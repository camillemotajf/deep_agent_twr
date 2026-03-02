from typing import List

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

from app.config.container import campaign_service, request_service
from app.utils.analise import analyze_frequencies


MODELS_PATH = f"G:/Meu Drive/TWR/data"
LABEL_MAP = {"bots": 0, "unsafe": 1, "bot": 0}
EMBEDDING_CONFIG = "fasttext"
TRANSFORMER_MODEL = "all-MiniLm-L6-v2"
FATSTEXT_PATH = f"{MODELS_PATH}/embedding"
BASE_MODEL_PATH = "files/models"


@tool
def run_ml_inference_pipeline() -> str:
    """
    EXECUTE FIRST. Runs MIL inference on data currently in Global Context.
    Evaluation is performed at BAG LEVEL (correct for MIL).
    """
    try:
        df = AnalysisContext.get_data_from_mongo()
        traffic_source = AnalysisContext.get_traffic_source()
    except ValueError as e:
        return f"Error: No data loaded. Ask the Orchestrator to load a file first. ({e})"

    if df.empty:
        return "Error: Dataset is empty."

    print(f"Starting MIL inference for traffic source: {traffic_source}")
    print(f"Embedding config: {EMBEDDING_CONFIG}")

    try:
        inference_service = ModelService(
            traffic_source=traffic_source,
            emb_config=EMBEDDING_CONFIG,
        )
    except Exception as e:
        return f"Error initializing ModelService: {e}"

    try:
        df_results = inference_service.predict(df)
        print("Inference sample:")
        print(df_results[["ip", "bag_id", "pred", "certeza_bag"]].head())
    except Exception as e:
        return f"Error during ML inference execution: {e}"
    
    # ---------------------------------------------------------
    # ETAPA E: AVALIAÇÃO DE DESEMPENHO (MÉTRICAS)
    # ---------------------------------------------------------
    acuracia = (df_results["decision_mil"] == df_results["pred"]).mean()
    total_erros = (df_results["decision_mil"] != df_results["pred"]).sum()
    
    fp = len(df_results[(df_results["decision_mil"] == 0) & (df_results["pred"] == 1)])
    fn = len(df_results[(df_results["decision_mil"] == 1) & (df_results["pred"] == 0)])

    # # -------------------------------------------------
    # # ✅ CORRECT MIL EVALUATION (BAG LEVEL)
    # # -------------------------------------------------
    # if "decision" in df_results.columns:
    #     # Normalize decision labels
    #     df_results["decision_norm"] = (
    #         df_results["decision"]
    #         .str.lower()
    #         .replace({"bot": "bots"})
    #         .map({"bots": 1, "unsafe": 0})
    #     )

    #     # Build BAG-LEVEL evaluation table
    #     bag_eval_df = (
    #         df_results
    #         .groupby("bag_id")
    #         .agg(
    #             target=("decision_norm", "max"),  # MIL ground truth
    #             pred=("pred", "first"),
    #             bag_probability=("bag_probability", "first")
    #         )
    #         .reset_index()
    #     )

    #     bag_eval_df["is_error"] = bag_eval_df["target"] != bag_eval_df["pred"]

    #     accuracy = (bag_eval_df["target"] == bag_eval_df["pred"]).mean()
    #     total_errors = bag_eval_df["is_error"].sum()

    #     # False Positives / False Negatives (BAG LEVEL)
    #     qtd_fp = len(
    #         bag_eval_df[(bag_eval_df["target"] == 0) & (bag_eval_df["pred"] == 1)]
    #     )
    #     qtd_fn = len(
    #         bag_eval_df[(bag_eval_df["target"] == 1) & (bag_eval_df["pred"] == 0)]
    #     )
    # else:
    #     accuracy = 0.0
    #     total_errors = 0
    #     qtd_fp = 0
    #     qtd_fn = 0
    #     bag_eval_df = None

    # Save detailed (line-level) results for downstream tools
    AnalysisContext.set_ml_results_data(df_results)

    print(
        "Inference completed | "
        f"Lines: {len(df_results)} | "
        # f"Bags: {bag_eval_df.shape[0] if bag_eval_df is not None else 0}"
    )

    return (
        f"Inference completed using MIL '{traffic_source}' model.\n"
        f"- Total samples (lines): {len(df_results)}\n"
        # f"- Total bags evaluated: {bag_eval_df.shape[0] if bag_eval_df is not None else 0}\n"
        f"- Model Accuracy (bag-level): {acuracia * 100:.2f}%\n"
        f"- Total bag prediction errors: {total_erros}\n"
        f"- False Positives (real = unsafe | pred = bots): {fp}\n"
        f"- False Negatives (real = bots | pred = unsafe): {fn}\n\n"
        "Next steps:\n"
        "1. Call 'get_dataset_health_check' for bag-level stats.\n"
        "2. Call 'query_anomalous_ids' to inspect specific bags.\n"
        "3. Use attention weights to explain MIL decisions."
    )

@tool
async def get_context_data(
    excluded_hashes: List[str],
    traffic_source: str
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

    df_database = pd.DataFrame(requests)
    df_analysis = AnalysisContext.get_data_to_analise()

    if df_database.empty or df_analysis.empty:
        return {
            "status": "empty",
            "message": "Not enough data to run analysis"
        }

    frequencies = analyze_frequencies(
        df_analysis=df_analysis,
        df_database=df_database
    )

    strong_bot_signals = (
        frequencies[
            (frequencies["class"] == "bot") &
            (frequencies["value"] != "absent") &
            (frequencies["percentage"] > 70)
        ]
        .sort_values("percentage", ascending=False)
        .head(20)
    )

    return {
        "status": "ok",
        "ml_rows": len(df_analysis),
        "database_rows": len(df_database),
        "strong_bot_signals": strong_bot_signals.to_dict(orient="records")
    }



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
