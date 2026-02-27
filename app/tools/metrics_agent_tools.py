from langchain.tools import tool
import pandas as pd
import json
from bson import json_util
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
from app.services.embedding_service import EmbeddingService
from app.tools.context_store import AnalysisContext
from app.services.mil_ia_service import ModelService
from app.config.settings import settings


MODELS_PATH = f"G:/Meu Drive/TWR/data"
LABEL_MAP = {"bots": 0, "unsafe": 1, "bot": 0}
EMBEDDING_CONFIG = "fasttext"
TRANSFORMER_MODEL = "all-MiniLm-L6-v2"
FATSTEXT_PATH = f"{MODELS_PATH}/embedding"
BASE_MODEL_PATH = "files/models"


@tool
def run_ml_inference_pipeline() -> str:
    """
    EXECUTE FIRST. Runs ML inference on data currently in Global Context.
    PREREQUISITE: Orchestrator must have loaded data.
    ARGS: None.
    RETURNS: Summary string. Updates internal state for queries.
    """
    try:
        df = AnalysisContext.get_data_from_mongo()
        traffic_source = AnalysisContext.get_traffic_source()
    except ValueError as e:
        return f"Error: No data loaded. Ask the Orchestrator to load a file first. ({e})"

    if df.empty:
        return "Error: Dataset is empty."
    
    print(f"Iniciando Inferência MIL para a fonte: {traffic_source}")
    print(f"Embedding type: {EMBEDDING_CONFIG}")

    try:
        inference_service = ModelService(
            traffic_source=traffic_source,
            emb_config=EMBEDDING_CONFIG,
        )
    except Exception as e:
        return f"Erro ao chamar o ModelService: {e}"

    try:
        # Se o seu método no serviço ainda tiver 'async def', adicione o 'await' antes de inference_service
        df_results = inference_service.predict(df)
        print("Amostra dos resultados:")
        print(df_results[['ip', 'bag_id', 'pred', 'certeza_bag']].head())
    except Exception as e:
        return f"Error during ML inference execution: {e}"
    
    # === AQUI ESTÁ A MUDANÇA CRUCIAL ===
    if "decision" in df_results.columns:
        # 1. Padroniza o texto do banco e converte para números (1 = bots, 0 = unsafe)
        df_results["target"] = df_results["decision"].str.lower().replace({"bot": "bots"}).map({"bots": 1, "unsafe": 0})
        
        df_results["is_error"] = (df_results["target"] != df_results["pred"])
        accuracy = (df_results["target"] == df_results["pred"]).mean()
        total_errors = df_results["is_error"].sum()
        
        # 3. FP e FN também usam a lógica matemática (1 e 0)
        # Falso Positivo: Era humano (0), mas o modelo disse Bot (1)
        df_fp = df_results[(df_results["target"] == 0) & (df_results["pred"] == 1)]
        # Falso Negativo: Era bot (1), mas o modelo disse Humano (0)
        df_fn = df_results[(df_results["target"] == 1) & (df_results["pred"] == 0)]
        
        qtd_fp = len(df_fp)
        qtd_fn = len(df_fn)
    else:
        df_results["is_error"] = False
        accuracy = 0.0
        total_errors = 0
        qtd_fp = 0
        qtd_fn = 0

    # Salva o dataframe processado no contexto
    AnalysisContext.set_ml_results_data(df_results)

    print(f"Checando se a tool de inferência salva os dados: {len(AnalysisContext.get_data_to_analise())}")

    return (
        f"Inference completed using MIL '{traffic_source}' model.\n"
        f"- Analyzed: {len(df_results)} samples.\n"
        f"- Model Accuracy: {accuracy * 100:.2f}%\n"
        f"- Total prediction discrepancies: {total_errors}\n"
        f"- Total False Positives (real = unsafe | pred = bots): {qtd_fp}\n"
        f"- Total False Negatives (real = bots | pred = unsafe): {qtd_fn}\n\n"
        "You can now:\n"
        "1. Call 'get_dataset_health_check' to see overall performance stats.\n"
        "2. Call 'query_anomalous_ids' to extract specific samples for the Detective Agent."
    )


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
