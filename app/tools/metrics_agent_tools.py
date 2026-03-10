from typing import List

from langchain.tools import tool
import pandas as pd
import json
from bson import json_util
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader
import os
import seaborn as sns
import matplotlib.pyplot as plt

from app.mentor_net.mentor_preditor import MentorNetPredictor
from app.services import ml_noise_service
from app.mentor_net.mentornet import MentorNet
from app.mentor_net.student_mlp import MLPStudent
from app.mentor_net.trainer import Trainer
from app.mentor_net.history_buffer import HistoryBuffer
from app.mentor_net.http_data import HTTPLogDataset
from app.services import clustering_service
from app.services.attetion_mil_article import MILAttetionService
from app.services.clustering_service import RequestClusteringPipeline
from app.services.embedding_service import EmbeddingService
from app.tools.context_store import AnalysisContext
from app.services.mil_ia_service import ModelService
from app.config.settings import settings

from app.config.container import campaign_service, request_service
from app.utils.analise import analyze_frequencies


MODELS_PATH = f"G:/Meu Drive/TWR/data"
LABEL_MAP = {"bots": 0, "unsafe": 1, "bot": 0}
EMBEDDING_CONFIG = "fasttext"
TRANSFORMER_MODEL = "all-MiniLm-L6-v2"
FATSTEXT_PATH = f"{MODELS_PATH}/embedding"
BASE_MODEL_PATH = "files/models"

model = SentenceTransformer('all-MiniLM-L6-v2')
clustering_service = RequestClusteringPipeline(
      embedding_model=model,
      max_header_freq=0.7,
      min_cluster_size=30
)

@tool
def analyze_traffic_patterns(filepath: str) -> str:
    """
    Ferramenta de Machine Learning especializada em encontrar botnets e ataques.
    Use esta ferramenta passando APENAS o caminho do arquivo parquet (.parquet).
    Ela retornará um JSON classificando o tráfego em 'bot', 'unsafe', 'mixed' ou 'noise_anomaly'.
    """
    # A ferramenta apenas repassa o caminho para o método do serviço
    return clustering_service.analyze_traffic_for_llm(filepath)

@tool
def run_ml_inference_pipeline(file_path: str, traffic_source: str) -> str:
    """
    MUST BE CALLED SECOND (after data extraction).
    Runs the Multiple Instance Learning (MIL) inference pipeline on the raw dataset.
    
    Args:
        file_path: The exact path to the .parquet file containing the raw HTTP requests.
        traffic_source: The source domain (e.g., 'google', 'tiktok') to load the correct embedding model.
        
    Returns:
        A string containing the inference metrics (Accuracy, False Positives, False Negatives) 
        and the 'results_path' pointing to the newly generated .parquet file with predictions.
        You MUST pass this 'results_path' to the Data Analyst agent for mismatch investigation.
    """
    try:
        df = pd.read_parquet(file_path)
    except ValueError as e:
        return f"Error: No data loaded. Ask the Orchestrator to load a file first. ({e})"

    if df.empty:
        return "Error: Dataset is empty."

    try:
        inference_service = MILAttetionService(
            traffic_source=traffic_source,
            emb_config=EMBEDDING_CONFIG,
        )
    except Exception as e:
        return f"Error initializing ModelService: {e}"

    try:
        df_results = inference_service.predict(df)
        results_path = file_path.replace("raw_", "results_")
        df_results.to_parquet(results_path)

    except Exception as e:
        return f"Error during ML inference execution: {e}"
    

    acuracia = (df_results["decision_mil"] == df_results["mil_prediction"]).mean()
    total_erros = (df_results["decision_mil"] != df_results["mil_prediction"]).sum()
    
    fp = len(df_results[(df_results["decision_mil"] == 0) & (df_results["mil_prediction"] == 1)])
    fn = len(df_results[(df_results["decision_mil"] == 1) & (df_results["mil_prediction"] == 0)])

    return (
        f"SUCCESS: MIL Inference completed for '{traffic_source}'.\n"
        f"- Total samples evaluated: {len(df_results)}\n"
        f"- Model Accuracy: {acuracia * 100:.2f}%\n"
        f"- Total prediction errors: {total_erros}\n"
        f"- False Positives (real = unsafe | pred = bots): {fp}\n"
        f"- False Negatives (real = bots | pred = unsafe): {fn}\n\n"
        f"CRITICAL: The prediction results were saved to: {results_path}\n"
        "Action Required: Pass this file path and the metrics to the 'bot-data-analyst' so they can investigate the False Positives and False Negatives."
    )

@tool
def clear_directory(file_path, predictions_file_path):
    """
    Clears the temporaries paths that have been used for analysis 
    Called at the end of workflow of th data analyst
    Args:
        file_path: The exact path to the .parquet file containing the raw HTTP requests.
        predictions_file_path: The .parquet file containing the ML predictions.
    """

    os.remove(file_path)
    os.remove(predictions_file_path)

    return {
        "status": "sucess",
        "message": "temporary files already deleted"
    }
    
@tool
async def compare_mismatch_frequencies(
    predictions_file_path: str, 
    excluded_hashes: List[str], 
    traffic_source: str,
    mismatch_type: str
) -> str:
    """
    Analyzes the frequency of HTTP headers and URL parameters to find bot patterns.
    
    Args:
        predictions_file_path: The .parquet file containing the ML predictions.
        excluded_hashes: List of campaign hashes currently being analyzed by the ML Agent (to exclude from baseline).
        traffic_source: The source domain (e.g., 'google', 'tiktok').
        mismatch_type: Must be either "FP" (False Positives: real=unsafe, pred=bots) or "FN" (False Negatives: real=bots, pred=unsafe).
        
    Returns:
        A JSON string detailing the most anomalous header and parameter keys/values compared to a baseline.
    """
    try:
        df_analysis = pd.read_parquet(predictions_file_path)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error loading predictions file: {e}"})

    if mismatch_type == "FP":
        df_focus = df_analysis[(df_analysis["decision_mil"] == 0) & (df_analysis["mil_prediction"] == 1)]
    elif mismatch_type == "FN":
        df_focus = df_analysis[(df_analysis["decision_mil"] == 1) & (df_analysis["mil_prediction"] == 0)]
    else:
        return json.dumps({'status': 'error', "message": f"Error: mismatch_type must be 'FP' or 'FN'."})

    if df_focus.empty:
        return json.dumps({"status": "error", "message": f"No {mismatch_type} mismatches found to analyze."})

    hashes = await campaign_service.fetch_recent_active_campaign_hashes_excluding(
        excluded_hashes=excluded_hashes, traffic_source=traffic_source
    )
    requests_baseline = await request_service.fetch_training_sample_by_hashes(
        hashes=hashes, limit_each=1000
    )
    df_baseline = pd.DataFrame(requests_baseline)

    print("Baseline: ", df_baseline.head())
    print("Analise: ", df_focus.head())

    df_focus_renamed = df_focus.rename(columns={
        "mil_prediction": "pred",
        "decision_mil": "target"   
    })

    frequencies = analyze_frequencies(df_analysis=df_focus_renamed, df_database=df_baseline)
    print("Frequências: ", frequencies)

    # 🚀 O Coração Matemático
    frequencies = analyze_frequencies(df_analysis=df_focus_renamed, df_database=df_baseline)
    
    # ... (início da Tool continua igual, chamando analyze_frequencies) ...
    
    df_ml = frequencies[frequencies["source"] == "ml"].copy()
    df_db = frequencies[frequencies["source"] == "database"].copy()

    colunas_chave = ['feature_type', 'key', 'value']
    colunas_metricas = ['count_bots', 'count_unsafe', 'pct_bots', 'pct_unsafe']
    
    df_ml = df_ml[colunas_chave + colunas_metricas]
    df_db = df_db[colunas_chave + colunas_metricas]

    # Cruzando as Predições do ML com a Realidade do Banco
    comparison = pd.merge(
        df_ml, 
        df_db, 
        on=colunas_chave, 
        how='left', 
        suffixes=('_mismatch', '_baseline')
    )

    # Limpando Nulos
    for col in ['count_bots_baseline', 'count_unsafe_baseline', 'pct_bots_baseline', 'pct_unsafe_baseline']:
        comparison[col] = comparison[col].fillna(0)

    # A Vassoura Anti-Lixo
    chaves_proibidas = ['pred', 'target', 'decision', 'decision_mil', 'mil_prediction', 'is_error', 'id', 'bag_id', 'embedding', 'attention_weight']
    comparison = comparison[~comparison['key'].isin(chaves_proibidas)]

    # Métrica de rank: Quantas vezes o erro apareceu no total (bots + unsafe)?
    comparison["sort_metric"] = comparison["count_bots_mismatch"] + comparison["count_unsafe_mismatch"]
    strong_signals = comparison.sort_values(by='sort_metric', ascending=False).head(15)

    # 🚀 O EMPACOTAMENTO DOS DADOS (O que você pediu!)
    def format_occurrences(row, suffix):
        return {
            "bots": {
                "count": int(row[f"count_bots_{suffix}"]), 
                "percentage": round(row[f"pct_bots_{suffix}"], 2)
            },
            "unsafe": {
                "count": int(row[f"count_unsafe_{suffix}"]), 
                "percentage": round(row[f"pct_unsafe_{suffix}"], 2)
            }
        }

    strong_signals["total_occurrences_mismatch"] = strong_signals.apply(lambda x: format_occurrences(x, "mismatch"), axis=1)
    strong_signals["total_occurrences_baseline"] = strong_signals.apply(lambda x: format_occurrences(x, "baseline"), axis=1)

    # Filtrando apenas as colunas limpas para enviar para o JSON
    colunas_finais_json = [
        "feature_type", "key", "value", 
        "total_occurrences_mismatch", 
        "total_occurrences_baseline"
    ]
    
    final_data = strong_signals[colunas_finais_json].to_dict(orient="records")

    return json.dumps({
        "status": "success",
        "mismatch_type": mismatch_type,
        "anomalous_patterns_comparison": final_data,
        "hint": "Each feature shows its occurrence inside 'bots' and 'unsafe' classes individually. Compare mismatch percentages against baseline percentages to conclude if the ML hallucinated or if the human label was wrong."
    }, indent=2)


@tool
async def get_context_data(
    excluded_hashes: List[str],
    traffic_source: str,
    data_result_path: str,
    mismatch_type: str
) -> dict:
    """
    Fetches contextual traffic data, runs frequency analysis,
    and returns explainable signals for the agent.
    """

    hashes = await campaign_service.fetch_recent_active_campaign_hashes_excluding(
        excluded_hashes=excluded_hashes,
        traffic_source=traffic_source
    )

    requests = await request_service.fetch_training_sample_by_hashes(
        hashes=hashes,
        limit_each=1000
    )

    df_baseline = pd.DataFrame(requests)
    df_analysis = pd.read_parquet(data_result_path)

    df_focus_renamed = df_analysis.rename(columns={
        "mil_prediction": "pred",
        "decision_mil": "target"
    })

    frequencies = analyze_frequencies(df_analysis=df_focus_renamed, df_database=df_baseline)
    
    df_ml = frequencies[frequencies["source"] == "ml"].copy()
    df_db = frequencies[frequencies["source"] == "database"].copy()

    colunas_chave = ['feature_type', 'key', 'value']
    colunas_metricas = ['total_occurrences', 'pct_bots', 'pct_unsafe']
    
    df_ml = df_ml[colunas_chave + colunas_metricas]
    df_db = df_db[colunas_chave + colunas_metricas]

    # Cruzamos as predições do ML com a Realidade Histórica do Banco
    comparison = pd.merge(
        df_ml, 
        df_db, 
        on=colunas_chave, 
        how='left', 
        suffixes=('_mismatch', '_baseline')
    )

    # Limpando Nulos e Lixo
    for col in ['total_occurrences_baseline', 'pct_bots_baseline', 'pct_unsafe_baseline']:
        comparison[col] = comparison[col].fillna(0)

    chaves_proibidas = ['pred', 'target', 'decision', 'decision_mil', 'mil_prediction', 'is_error', 'id', 'bag_id', 'embedding', 'attention_weight']
    comparison = comparison[~comparison['key'].isin(chaves_proibidas)]

    # Filtramos para pegar os padrões que apareceram mais vezes nos erros
    strong_signals = comparison.sort_values(by='total_occurrences_mismatch', ascending=False).head(15)

    # Arredondamento para não gastar tokens
    for col in ['pct_bots_mismatch', 'pct_unsafe_mismatch', 'pct_bots_baseline', 'pct_unsafe_baseline']:
        strong_signals[col] = strong_signals[col].round(1)

    return json.dumps({
        "status": "success",
        "mismatch_type": mismatch_type,
        "anomalous_patterns_comparison": strong_signals.to_dict(orient="records"),
        "hint": "If mismatch_type=FP (model said bot), and pct_bots_baseline is HIGH, the human label was wrong. If pct_bots_baseline is LOW, the ML model hallucinated."
    }, indent=2)



@tool
def get_dataset_health_check() -> dict:
    """
    GET DIAGNOSTICS. Use after inference.
    RETURNS: JSON with {total_samples, false_positives, false_negatives, avg_trust}.
    """
    df = AnalysisContext.get_data_to_analise()
    return {
        "total_samples": len(df),
        "false_positives": int(((df.target==0) & (df.pred==1)).sum()),
        "false_negatives": int(((df.target==1) & (df.pred==0)).sum()),
        "avg_trust": float(df["weight"].mean())
    }

@tool
def query_anomalous_ids(criteria: str, threshold: float = 0.5) -> list[int]:
    """
    GET SUSPICIOUS IDs. Returns top 50 int IDs for investigation.
    ARGS:
    - criteria: 'low_trust' (noisy labels), 'high_loss' (hard samples), 'disagreement' (prediction errors).
    - threshold: Cutoff float (default 0.5).
    """
    df = AnalysisContext.get_data_to_analise()
    
    if criteria == "low_trust":
        subset = df[df["weight"] < threshold]
    elif criteria == "high_loss":
        subset = df[df["loss"] > threshold]
    elif criteria == "disagreement":
        subset = df[df["target"] != df["pred"]]
    else:
        return []

    return subset["id"].head(50).tolist()
