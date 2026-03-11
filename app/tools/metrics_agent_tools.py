import ast
from typing import List, Optional, Dict, Any
from langchain.tools import tool
import pandas as pd
import json
# from sentence_transformers import SentenceTransformer
# from torch.utils.data import DataLoader
import os

# from app.mentor_net.mentor_preditor import MentorNetPredictor
# from app.mentor_net.mentornet import MentorNet
# from app.mentor_net.student_mlp import MLPStudent
# from app.mentor_net.trainer import Trainer
# from app.mentor_net.history_buffer import HistoryBuffer
# from app.mentor_net.http_data import HTTPLogDataset
from app.services.attetion_mil_article import MILAttetionService
from app.services.clustering_service import RequestClusteringPipeline
# from app.services.embedding_service import EmbeddingService
from app.tools.context_store import AnalysisContext
# from app.services.mil_ia_service import ModelService
from app.config.settings import settings
from langchain_community.tools import TavilySearchResults

from app.config.container import campaign_service, request_service
from app.utils.analise import analyze_frequencies, parse_dict_col
tavily_search = TavilySearchResults(max_results=1)


# MODELS_PATH = f"G:/Meu Drive/TWR/data"
# LABEL_MAP = {"bots": 0, "unsafe": 1, "bot": 0}
# EMBEDDING_CONFIG = "fasttext"
# TRANSFORMER_MODEL = "all-MiniLm-L6-v2"
# FATSTEXT_PATH = f"{MODELS_PATH}/embedding"
# BASE_MODEL_PATH = "files/models"

clustering_service = RequestClusteringPipeline()

tavily_tool = TavilySearchResults(
    max_results=2, 
    description=(
        "A web search tool optimized for threat research, OSINT, and cybersecurity investigations. "
        "Use this tool to search the web for information on ISPs, User-Agents, IP reputations, and others Browsers headers pattern with"
        "malware signatures, and general threat intelligence based on the artifacts extracted."
    )
)

    
@tool()
def extract_suspicious_artifacts(
    filepath: str, 
    fields_to_extract: Optional[List[str]] = None, 
    n_values: int = 15,
    cluster_column: str = "classification" # NOME DA COLUNA DO SEU MODELO (ex: 'cluster', 'label', 'classification')
) -> str:
    """
    Reads the anomaly .parquet file and extracts the most frequent (Top N) values from specific HTTP headers, User-Agents, or ISPs.
    The Agent MUST use this tool first to gather the suspicious artifacts before utilizing web search tools for threat intelligence.

    Args:
        filepath (str): Path to the .parquet file containing the anomalous HTTP requests.
        fields_to_extract (Optional[List[str]]): List of specific fields or headers to extract (e.g., ["user-agent", "ip_api_isp", "x-body-platform"]). Defaults to ["user-agent", "ip_api_isp"] if not provided.
        n_values (int): The number of top most frequent values to return per field. Default is 3.

    Returns:
        str: A JSON-formatted string containing the extracted top values for each requested field.
    """
    
    if not fields_to_extract:
        fields_to_extract = ["user-agent", "ip_api_isp", "sec-fetch-dest", "sec-fetch-site"]
        
    try:
        df = pd.read_parquet(filepath)
        
        # 1. Reidratação da coluna 'headers'
        if "headers" in df.columns:
            def safe_parse(val):
                if isinstance(val, dict): return val
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: 
                        try: return ast.literal_eval(val)
                        except: return {}
                return {}
            df["headers"] = df["headers"].apply(safe_parse)

        # 2. Função interna para extrair os Top N de um "pedaço" do DataFrame
        def get_top_artifacts(sub_df):
            extracted_data = {}
            for field in fields_to_extract:
                field_lower = field.lower()
                extracted_vals = []
                
                df_cols_lower = {str(c).lower(): c for c in sub_df.columns}
                if field_lower in df_cols_lower:
                    real_col = df_cols_lower[field_lower]
                    top_values = sub_df[real_col].dropna().value_counts().head(n_values).index.tolist()
                    extracted_vals = [str(v).strip() for v in top_values if str(v).strip().lower() not in ["none", "nan", "unknown", ""]]
                
                elif "headers" in sub_df.columns:
                    extracted = []
                    for h_dict in sub_df["headers"]:
                        if isinstance(h_dict, dict):
                            h_norm = {str(k).lower(): str(v).strip() for k, v in h_dict.items()}
                            val = h_norm.get(field_lower) 
                            if val: extracted.append(val)
                    if extracted:
                        top_values = pd.Series(extracted).value_counts().head(n_values).index.tolist()
                        extracted_vals = [str(v).strip() for v in top_values if str(v).strip().lower() not in ["none", "nan", "unknown", ""]]
                
                if extracted_vals:
                    extracted_data[field] = extracted_vals
                    
            return extracted_data

        # 3. O Pulo do Gato: Agrupando por Cluster!
        final_payload = {}
        
        # Se a coluna de classificação do cluster existir no DataFrame:
        if cluster_column in df.columns:
            # Agrupa os dados pelo nome do cluster e extrai separadamente
            for cluster_name, group_df in df.groupby(cluster_column):
                
                # Ignorar clusters que são 100% corretos para economizar tokens (opcional)
                if cluster_name in ["correct_unsafe", "correct_bot"]:
                    continue 

                artifacts = get_top_artifacts(group_df)
                if artifacts:
                    final_payload[str(cluster_name)] = artifacts
        else:
            # Fallback seguro: se não achar a coluna, bota tudo no unknown_cluster
            artifacts = get_top_artifacts(df)
            if artifacts:
                final_payload["unknown_cluster"] = artifacts
                
        output_filepath = filepath.replace(".parquet", "_artifacts.json")
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)
            
        return json.dumps({
            "status": "success", 
            "message": "Artefatos extraídos e salvos com sucesso.",
            "extracted_filepath": output_filepath
        })
        
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {str(e)}"})
        

@tool
def analyze_traffic_patterns(filepath: str) -> str:
    """
    Machine Learning tool specialized in detecting botnets and attacks.
    Use this tool by passing ONLY the path to the parquet file (.parquet).
    It will return a JSON classifying the traffic as 'bot', 'unsafe', 'mixed', or 'noise_anomaly'.
    """
    clustering_service.run(filepath)
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
            emb_config=settings.EMBEDDING_CONFIG,
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

import json
import ast
import pandas as pd
from typing import Dict, List, Optional, Any
from langchain.tools import tool

@tool()
async def filter_artifacts_by_baseline(
    extracted_filepath: str, 
    hashes_to_exclude: List[str],
    traffic_source: Optional[str] = None
) -> str:
    """
    Recebe os artefatos suspeitos (do cluster) e os valida contra o histórico (baseline).
    Descarta valores que são comprovadamente tráfego humano (Falsos Positivos) e retorna APENAS os alvos para OSINT.
    """
    try:
        # ==========================================
        # 1. TRATAMENTO DO FORMATO DO LLM
        # ==========================================
        # Transforma o formato plano recebido {'user-agent': ['...'], 'ip_api_isp': ['...']} 
        # em um formato padronizado para o nosso loop funcionar.
        # Blindagem: se o LLM mandar um dicionário com o path dentro
        if isinstance(extracted_filepath, dict):
            extracted_filepath = extracted_filepath.get("extracted_filepath", "") or extracted_filepath.get("filepath", "")
        extracted_filepath = str(extracted_filepath).strip()

        # Verifica se o arquivo realmente existe
        if not os.path.exists(extracted_filepath):
            return json.dumps({"status": "error", "message": f"Arquivo não encontrado: {extracted_filepath}"})

        # ==========================================
        # LENDO O PAYLOAD DO DISCO (CUSTO ZERO DE TOKENS)
        # ==========================================
        with open(extracted_filepath, "r", encoding="utf-8") as f:
            suspicious_artifacts = json.load(f)

        # ==========================================
        # 2. BUSCA NO BANCO DE DADOS
        # ==========================================
        if not traffic_source:
            traffic_source = await campaign_service.fetch_traffic_source_by_hash(hashes_to_exclude[0])

        hashes_baseline = await campaign_service.fetch_recent_active_campaign_hashes_excluding(
            traffic_source=traffic_source, excluded_hashes=hashes_to_exclude, limit=50
        )

        print("Hashes baseline: ", hashes_baseline)
        print("Traffic source: ", traffic_source)

        request_baseline = await request_service.fetch_training_sample_by_hashes(hashes=hashes_baseline, limit_each=1000)
        print("request baseline len: ", len(request_baseline))

        df_baseline = pd.DataFrame(request_baseline)
        
        if "decision" not in df_baseline.columns:
            return json.dumps({"status": "error", "message": "Baseline não possui a coluna 'decision'."})

        # Prepara a coluna headers do baseline se for string
        if "headers" in df_baseline.columns:
            def safe_parse(val):
                if isinstance(val, dict): return val
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: 
                        try: return ast.literal_eval(val)
                        except: return {}
                return {}
            df_baseline["headers"] = df_baseline["headers"].apply(safe_parse)

        report = {
            "status": "success",
            "APPROVED_FOR_OSINT": {},
            "REJECTED_FALSE_POSITIVES": {}
        }

        # 2. Cruza os valores iterando pelos CLUSTERS fornecidos
        for cluster_name, fields_dict in suspicious_artifacts.items():
            report["APPROVED_FOR_OSINT"][cluster_name] = {}
            report["REJECTED_FALSE_POSITIVES"][cluster_name] = {}

            for field, values in fields_dict.items():
                field_lower = field.lower()
                report["APPROVED_FOR_OSINT"][cluster_name][field] = {}
                report["REJECTED_FALSE_POSITIVES"][cluster_name][field] = {}

                for val in values:
                    val_lower = str(val).lower().strip()
                    matches = pd.Series(False, index=df_baseline.index)
                    
                    # Busca o valor dentro dos headers do baseline
                    if "headers" in df_baseline.columns:
                        def check_match(h_dict):
                            if not isinstance(h_dict, dict): return False
                            # Converte chaves para minúsculo para garantir o match
                            h_lower_keys = {str(k).lower(): str(v).strip().lower() for k, v in h_dict.items()}
                            return h_lower_keys.get(field_lower) == val_lower
                        
                        matches = df_baseline["headers"].apply(check_match)

                    subset = df_baseline[matches]
                    total_occurrences = len(subset)
                    
                    # Se não tem histórico suficiente, é suspeito por ser novidade (Zero-Day)
                    if total_occurrences < 10:
                        report["REJECTED_FALSE_POSITIVES"][cluster_name][field][val] = f"Sem dados no baseline para análise"
                        continue

                    # Calcula a distribuição percentual
                    dist = subset["decision"].value_counts(normalize=True) * 100
                    pct_bot = dist.get("bots", 0.0)
                    pct_human = dist.get("unsafe", 0.0)
                    
                    # ==========================================
                    # LÓGICA DE TRIAGEM POR TIPO DE CLUSTER
                    # ==========================================
                    if cluster_name == "unsafe_pred_bot":
                        # REGRA 1: Tráfego humano suspeito de ser bot escondido.
                        # Só investigar na web se historicamente >= 70% forem BOTS reais.
                        if pct_bot >= 70.0:
                            report["APPROVED_FOR_OSINT"][cluster_name][field][val] = f"Aprovado. Alta chance de bot escondido ({pct_bot:.1f}% bot na baseline)."
                        else:
                            report["REJECTED_FALSE_POSITIVES"][cluster_name][field][val] = f"Descartado. Historicamente predominante humano ({pct_human:.1f}% humano)."
                            
                    elif cluster_name == "bot_pred_unsafe":
                        # REGRA 2: Tráfego bot classificado como humano.
                        # Só investigar na web se historicamente >= 70% forem HUMANOS reais (para evitar bloqueio injusto).
                        if pct_human >= 70.0:
                            report["APPROVED_FOR_OSINT"][cluster_name][field][val] = f"Aprovado. Verificar falso positivo / bloqueio injusto ({pct_human:.1f}% humano na baseline)."
                        else:
                            report["REJECTED_FALSE_POSITIVES"][cluster_name][field][val] = f"Descartado. O bloqueio original fazia sentido ({pct_bot:.1f}% bot)."
                            
                    else:
                        # REGRA PADRÃO (para 'mixed', 'noise_anomaly', etc)
                        if pct_human >= 85.0:
                            report["REJECTED_FALSE_POSITIVES"][cluster_name][field][val] = f"Descartado. Tráfego benigno ({pct_human:.1f}% humano)."
                        else:
                            report["APPROVED_FOR_OSINT"][cluster_name][field][val] = f"Aprovado. Tráfego suspeito ({pct_bot:.1f}% bot)."

        return json.dumps(report, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "fatal_error", "message": f"{type(e).__name__}: {str(e)}"})
    
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
