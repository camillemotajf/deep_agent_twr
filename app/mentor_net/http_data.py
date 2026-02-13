import ast
import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder # LabelEncoder fica em preprocessing, não calibration

class HTTPLogDataset(Dataset):
    def __init__(self, dataframe, label_map: dict, embeddings, texts, device="cpu"):
        
        self.df = dataframe.reset_index(drop=True)
        self.texts = texts
        self.embeddings = embeddings

        self.label_map = label_map
        self.targets = torch.tensor(
            [self.label_map[l] for l in self.df["decision"]], 
            dtype=torch.long
        )
        
        self.ids = torch.arange(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "x": self.embeddings[idx],     # Embedding pré-calculado
            "label": self.targets[idx],    # Label numérico
            "raw_text": self.texts[idx]    # Texto já limpo (pegamos da lista em cache)
        }