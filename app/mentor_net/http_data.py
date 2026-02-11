import ast
import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder # LabelEncoder fica em preprocessing, não calibration

class HTTPLogDataset(Dataset):
    def __init__(self, dataframe, label_map: dict, embeddings=None, model_name="all-MiniLM-L6-v2", device="cpu"):
        
        self.df = dataframe.reset_index(drop=True)
        self.texts = [self._serialize_row(row) for _, row in self.df.iterrows()]
        
        if embeddings is None:
            self.encoder = SentenceTransformer(model_name)
            self.embeddings = self.encoder.encode(
                self.texts,
                batch_size=64,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=device
            ).cpu()
        else:
            self.embeddings = embeddings

        self.label_map = label_map
        self.targets = torch.tensor(
            [self.label_map[l] for l in self.df["decision"]], 
            dtype=torch.long
        )
        
        self.ids = torch.arange(len(self.df))

    def clean_to_dict(self, val):
        if isinstance(val, dict): return val
        if isinstance(val, str):
            try: return ast.literal_eval(val)
            except: return {}
        return {}

    def _serialize_row(self, row):
        headers = self.clean_to_dict(row.get("headers", {}))
        params = self.clean_to_dict(row.get("request", {}))

        hash_patterns = [
            r'^[a-fA-F0-9]{32,}$',
            r'^[a-fA-F0-9-]{36}$',
            r'^[a-zA-Z0-9+/=]{50,}$',
            r'^eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+$'
        ]

        ua = headers.get('User-Agent') or headers.get('user-agent') or "Unknown"

        clean_params = []
        if isinstance(params, dict):
            for k, v in params.items():
                v_str = str(v).strip()
                if not any(re.match(p, v_str) for p in hash_patterns):
                    clean_params.append(f"{k}={v}")
        
        params_str = " ".join(clean_params) if clean_params else "no_params"

        other_headers = []
        keys_to_ignore = ['User-Agent', 'user-agent', 'Cookie', 'Authorization']
        for k, v in headers.items():
            if k not in keys_to_ignore:
                other_headers.append(f"{k}:{v}")
        
        return f"UA: {ua} | Params: {params_str} | Headers: {' '.join(other_headers)}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "x": self.embeddings[idx],     # Embedding pré-calculado
            "label": self.targets[idx],    # Label numérico
            "raw_text": self.texts[idx]    # Texto já limpo (pegamos da lista em cache)
        }