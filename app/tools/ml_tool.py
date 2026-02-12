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
from app.tools.context_store import AnalysisContext


MODELS_PATH = f"{os.getcwd()}/files"
LABEL_MAP = {"bots": 0, "unsafe": 1}


def _load_trainer(traffic_source: str, num_samples: int):
    model_path = f"{MODELS_PATH}/{traffic_source}"

    buffer = HistoryBuffer(
        num_samples=num_samples,
        window_size=10,
        feature_dim=3,
    )

    mentor = MentorNet()
    mentor.load_state_dict(torch.load(model_path, map_location="cpu"))
    mentor.eval()

    student = MLPStudent()
    student.load_state_dict(torch.load(model_path, map_location="cpu"))
    student.eval()

    return Trainer(
        student=student,
        mentor=mentor,
        history_buffer=buffer,
        device="cpu" 
    )


@tool
async def run_ml_inference(
    traffic_source: str,
) -> dict:
    """
    Executes MentorNet/Student ML inference to detect label noise and hidden bots.
    
    :param traffic_source: The traffic origin (e.g., 'google') to load specific model weights.
    :type traffic_source: str
    :param file_path: Path to a JSON file containing requests.
    :type file_path: str | None
    :return: summary of misclassifications, trust scores, and potential anomalies.
    :rtype: dict
    """

    # data = []

    # if file_path and os.path.exists(file_path):
    #     with open(file_path, "r") as f:
    #         data = json_util.loads(f.read())

    # df = pd.DataFrame(data)

    # if df.empty:
    #     return {"error": "Empty Dataframe to analise"}

    df = AnalysisContext.get_data_to_analise()
    
    dataset = HTTPLogDataset(df, label_map=LABEL_MAP)

    model_path = f"{MODELS_PATH}/{traffic_source}"

    mentor_predictor = MentorNetPredictor(artifact_path=model_path)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    df_results = mentor_predictor.predict(loader)


    summary_stats = {
        "traffic_source": traffic_source,
        "processed_samples": len(df),
        "misclassifications_detected": sum(1 for r in df_results if r['target'] != r['pred']),
        "average_loss": sum(r['loss'] for r in df_results) / len(df_results) if df_results else 0,
        "suspicious_ids": [r['id'] for r in df_results if r['weight'] < 0.5][:20]
    }

    return summary_stats

@tool
def summarize_misclassifications(df: pd.DataFrame) -> dict:
    """
    Summarizes false positives and false negatives.
    - false positives: true label = "bots" and pred label = "unsafe"
    - false negatives: true label = "unsafe" and pred label = "bots"
    """
    fp = df[(df.true_label == 0) & (df.model_pred == 1)]
    fn = df[(df.true_label == 1) & (df.model_pred == 0)]

    return {
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "fp_rate": len(fp) / len(df),
        "fn_rate": len(fn) / len(df),
    }

@tool
def find_low_trust_samples(
    df: pd.DataFrame,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Finds samples with low mentor trust or high loss.
    """
    return df[
        (df["mentor_weight"] < threshold) |
        (df["loss"] > df["loss"].quantile(0.9))
    ]

@tool
def analyze_user_agent_patterns(
    df: pd.DataFrame,
    min_count: int = 5
) -> list[dict]:
    """
    Finds repeating User-Agent patterns in misclassified samples.
    """
    ua_counts = (
        df[df["is_error"]]
        .groupby("clean_ua")
        .size()
        .reset_index(name="count")
    )

    return ua_counts[ua_counts["count"] >= min_count] \
        .sort_values("count", ascending=False) \
        .to_dict(orient="records")


@tool
def prepare_ml_dataframe(ml_results: dict) -> dict:
    """
    Normalizes ML results into a DataFrame-ready structure
    and computes derived fields.
    """
    df = pd.DataFrame(ml_results["results"])

    df["is_error"] = df["true_label"] != df["model_pred"]
    df["risk_score"] = (
        df["loss"] * (1 - df["mentor_weight"]) * (1 - df["ncs"])
    )

    return {
        "total": len(df),
        "error_rate": float(df["is_error"].mean()),
        "data": df.to_dict(orient="records")
    }

@tool
def plot_risk_score_distribution(df: pd.DataFrame) -> str:
    """
    Plot the distribution of risk scores across requests.

    This tool visualizes how risk scores are distributed in the dataset,
    helping identify skewness, multi-modal behavior, or extreme outliers.
    It is useful for assessing whether the risk scoring logic is overly
    conservative, permissive, or unstable.

    Expected input:
    - df: pandas DataFrame containing a numeric column named `risk_score`

    Output:
    - Displays a histogram with kernel density estimation (KDE)
    - Returns a confirmation message after rendering the plot

    Usage notes:
    - Intended for exploratory and diagnostic analysis
    - Should be called only after risk scores have been computed
    - Does not modify or return the input data
    """

    sns.histplot(df["risk_score"], kde=True)
    plt.title("Risk Score Distribution")
    plt.show()
    return "Risk score distribution plotted"


@tool
def plot_ml_diagnostics(ml_results: dict) -> str:
    """
    Generate diagnostic plots for machine learning inference results.

    This tool produces multiple visual diagnostics to help assess
    model behavior, confidence, and potential bias. It is designed
    to support post-inference analysis and human interpretation.

    The following plots are generated:
    1. Distribution of mentor confidence weights
    2. Scatter plot of loss vs mentor confidence, colored by correctness
    3. Boxplot of mentor confidence grouped by true label

    Expected input:
    - ml_results: dictionary containing a key `results`, which is a list
      of per-sample inference records with at least the following fields:
        - true_label
        - model_pred
        - loss
        - mentor_weight

    Output:
    - Renders three diagnostic plots using matplotlib/seaborn
    - Returns a confirmation message once plots are displayed

    Usage notes:
    - Intended for analysis and debugging, not automated decision-making
    - Should be used by analytical agents or humans, not during inference
    - Does not alter or persist any data
    """

    df = pd.DataFrame(ml_results["results"])
    df["correct"] = df["true_label"] == df["model_pred"]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    sns.histplot(df["mentor_weight"], kde=True, ax=axes[0])
    axes[0].set_title("Distribuição de Confiança do Mentor")

    sns.scatterplot(
        data=df,
        x="loss",
        y="mentor_weight",
        hue="correct",
        ax=axes[1]
    )
    axes[1].set_title("Loss vs Confiança")

    sns.boxplot(
        data=df,
        x="true_label",
        y="mentor_weight",
        ax=axes[2]
    )
    axes[2].set_title("Viés por Classe")

    plt.tight_layout()
    plt.show()

    return "Gráficos de diagnóstico gerados com sucesso."

@tool
def run_ml_noise_analysis() -> dict:
    """
    Run noise and label-consistency analysis on the current ML dataset.

    This tool executes a noise detection scan to identify suspicious,
    low-confidence, or potentially mislabeled samples. It leverages
    pre-configured loaders and datasets to compute global statistics
    and extract high-risk examples.

    The analysis focuses on:
    - Label inconsistency
    - Unstable samples
    - Potential annotation errors

    Output:
    - A dictionary with:
        - stats_context: aggregated statistics and summary metrics
        - sample_preview: a preview (up to 50 rows) of suspicious samples
          represented as dictionaries

    Usage notes:
    - Assumes that the ML dataset and loader are already initialized
    - Does not perform model training or inference
    - Intended to support data quality audits and human review
    """
    result = ml_noise_service.run_noise_scan(
        loader=ml_noise_service.loader,
        original_dataframe=ml_noise_service.original_dataframe
    )

    return {
        "stats_context": result["stats_context"],
        "sample_preview": result["suspicious_df"].head(50).to_dict()
    }
