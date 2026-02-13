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


MODELS_PATH = f"{os.getcwd()}/files/models"
LABEL_MAP = {"bots": 0, "unsafe": 1}
EMBEDDING_CONFIG = "fasttext"
TRANSFORMER_MODEL = "intfloat/e5-base-v2"
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
    
    print("Embedding type: ", EMBEDDING_CONFIG)
    if EMBEDDING_CONFIG == "fasttext":
        model_path = FATSTEXT_PATH
    else:
        model_path = TRANSFORMER_MODEL

    try:
        EmbeddingService.get_instance(EMBEDDING_CONFIG, model_path, traffic_source=traffic_source)
    except FileNotFoundError:
        return f"Error: Embedding model not found at {model_path}"
    
    embeddings_matrix, texts = EmbeddingService.process_and_encode(
        df
    )
    print(embeddings_matrix.shape)

    dataset = HTTPLogDataset(df, label_map=LABEL_MAP, embeddings=embeddings_matrix, texts=texts)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model_path = f"{MODELS_PATH}/{traffic_source}/mentor_net_bundle_{EMBEDDING_CONFIG}.pth"
    mentor_predictor = MentorNetPredictor(artifact_path=model_path)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    df_results = mentor_predictor.predict(loader)
    df_results["is_error"] = (df_results["target"] != df_results["pred"])
    
    AnalysisContext.set_ml_results_data(df_results)


    print(f"checando se a tool de inferencia salva dos dados: {len(AnalysisContext.get_data_to_analise())}")

    accuracy = (df_results["target"] == df_results["pred"]).mean()
    total_errors = df_results["is_error"].sum()

    return (
        f"Inference completed using '{traffic_source}' model with results: \n"
        f"Models Accuracy: {accuracy}"
        f"Total Error in prediction (possible anomalies): {total_errors}"
        f"Analyzed {len(df_results)} samples.\n"
        "You can now now:\n"
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
