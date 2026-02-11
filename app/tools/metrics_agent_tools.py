from langchain.tools import tool
import pandas as pd
import json
from bson import json_util
import torch
from torch.utils.data import DataLoader
import os
import seaborn as sns
import matplotlib.pyplot as plt

from app.services import ml_noise_service
from app.mentor_net.mentornet import MentorNet
from app.mentor_net.student_mlp import MLPStudent
from app.mentor_net.trainer import Trainer
from app.mentor_net.history_buffer import HistoryBuffer
from app.mentor_net.http_data import HTTPLogDataset
from app.tools.context_store import AnalysisContext


MODELS_PATH = f"{os.getcwd()}/files/models"
LABEL_MAP = {"bots": 0, "unsafe": 1}


def _load_trainer(traffic_source: str, num_samples: int):
    model_path = f"{MODELS_PATH}/{traffic_source}"

    buffer = HistoryBuffer(
        num_samples=num_samples,
        window_size=10,
        feature_dim=3,
    )

    num_classes = 2

    mentor = MentorNet(
        input_size=3,       # Importante: [loss, diff, ncs]
        hidden_size=32,
        num_classes=num_classes,
    )
    mentor.load_state_dict(torch.load(f"{model_path}/mentor.pt", map_location="cpu"))
    mentor.eval()

    embed_dim = 384 

    student = MLPStudent(
        input_size=embed_dim,
        hidden_size=256,
        output_size=num_classes
    )
    student.load_state_dict(torch.load(f'{model_path}/student.pt', map_location="cpu"))
    student.eval()

    return Trainer(
        student=student,
        mentor=mentor,
        history_buffer=buffer,
        device="cpu" 
    )

@tool
def check_context() -> str:
    """
    Checks if the data info get by orchestrator was corrected loaded
    """

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

    dataset = HTTPLogDataset(df, label_map=LABEL_MAP)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    try:
        trainer = _load_trainer(traffic_source=traffic_source, num_samples=len(dataset))
    except Exception as e:
        return f"Error loading model for {traffic_source}: {str(e)}"

    results = []
    with torch.no_grad():
        for batch in loader:
            print(batch)
            x = batch["x"]
            y = batch["label"]
            ids = batch["id"]
            
            logits, embeddings = trainer.student(x, return_embeddings=True)
            preds = logits.argmax(dim=1)
            raw_losses = trainer.student.get_individual_losses(logits, y)
            
            weights = torch.ones_like(raw_losses)
            if trainer.mentor:
                hist = trainer.history.get(ids)
                epoch_vec = torch.full((len(y),), 99)
                weights = trainer.mentor(hist, y, epoch_vec)
                weights = torch.clamp(weights.view(-1), 0, 1)

            ncs = trainer.get_neighborhood_score(embeddings, y, k=10)

            for i in range(len(ids)):
                results.append({
                    "id": int(ids[i]),
                    "true_label": int(y[i]),
                    "model_pred": int(preds[i]),
                    "loss": float(raw_losses[i]),
                    "mentor_weight": float(weights[i]),
                    "ncs": float(ncs[i]),
                    "headers": df.loc[df.index == int(ids[i]), 'headers'].values[0] if 'headers' in df.columns else "N/A"

                })

    result_df = pd.DataFrame(results)
    result_df["is_error"] = result_df["true_label"] != result_df["model_pred"]
    
    AnalysisContext.set_ml_results_data(result_df)


    print(f"checando se a tool de inferencia salva dos dados: {len(AnalysisContext._df_ml_results)}")

    accuracy = (result_df["true_label"] == result_df["model_pred"]).mean()
    total_errors = result_df["is_error"].sum()

    return (
        f"Inference completed using '{traffic_source}' model with results: \n"
        f"Models Accuracy: {accuracy}"
        f"Total Error in prediction (possible anomalies): {total_errors}"
        f"Analyzed {len(result_df)} samples.\n"
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
        "false_positives": int(((df.true_label==0) & (df.model_pred==1)).sum()),
        "false_negatives": int(((df.true_label==1) & (df.model_pred==0)).sum()),
        "avg_trust": float(df["mentor_weight"].mean())
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
        subset = df[df["mentor_weight"] < threshold]
    elif criteria == "high_loss":
        subset = df[df["loss"] > threshold]
    elif criteria == "disagreement":
        subset = df[df["true_label"] != df["model_pred"]]
    else:
        return []

    return subset["id"].head(50).tolist()
