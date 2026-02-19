from functools import lru_cache
import os
import ast
import re
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from gensim.models import FastText
from tqdm import tqdm
from app.utils.http_tokens_ft import create_X_embedding_ft, create_vocabulary


@lru_cache(maxsize=2)
def load_cached_embedder(model_path: str, emb_type: str):
    print(f"⏳ [CACHE MISS] Lendo Embedding do disco: {model_path}...")
    
    if emb_type == "fasttext":
        return FastText.load(model_path)
        
    elif emb_type == "transformers":
        return torch.load(model_path, weights_only=True) 
        
    else:
        raise ValueError(f"Tipo de embedding desconhecido: {emb_type}")
    
class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def vector_size(self) -> int:
        pass
    
    @abstractmethod
    def get_text(self):
        pass

class TransformerEmbedder(BaseEmbedder):
      def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
            tqdm.pandas()
            if device is None:
                  self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                  self.device = device
            
            self.model = SentenceTransformer(model_name, device=self.device)
            self.texts = None

      def _clean_to_dict(self, val):
            if isinstance(val, dict): return val
            if isinstance(val, str):
                  try: return ast.literal_eval(val)
                  except: return {}
            return {}

      def _extract_http_feature(self, row: pd.Series, headers_col='headers', request_col='request') -> str:
            headers = self._clean_to_dict(row.get(headers_col, {}))
            params = self._clean_to_dict(row.get(request_col, {}))

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
      

      def encode(self, df: pd.DataFrame) -> np.ndarray:
            print("Enter to Transformers encoder")
            df_copy = df.copy()

            ## retirar produção
            df_copy["combined_text"] = df_copy.progress_apply(
                  self._extract_http_feature,
                  axis=1
            )
            # df_copy["combined_text"] = df_copy.apply(self._extract_http_feature, axis=1)
            texts = np.array(df_copy["combined_text"].to_list())
            self.texts = texts

            return self.model.encode(
                  texts, 
                  batch_size=32, 
                  show_progress_bar=True, 
                  convert_to_numpy=True,
                  normalize_embeddings=True
            )
      
      def get_text(self) -> List[str]:
            return self.texts

      @property
      def vector_size(self) -> int:
            return self.model.get_sentence_embedding_dimension()

class FastTextEmbedder(BaseEmbedder):
      def __init__(self, model_path: str, traffic_source: str):
            if not os.path.exists(model_path):
                  raise FileNotFoundError(f"Fast Text model not found: {model_path}")
            
            self.model_path = f"{model_path}/fasttext_{traffic_source}.model"

            print(f'[DEBUG] Model Path FASTTEXT: {self.model_path}')
            # self.model = FastText.load(self.model_path)
            self.model = load_cached_embedder(self.model_path, emb_type="fasttext")
            self.wv = self.model.wv
            self.texts = None
            self.traffic_source = traffic_source

      
      def encode(self, df: pd.DataFrame, traffic_source=None) -> np.ndarray:
            print("Enter to Fasttext encoder")
            corpus = create_vocabulary(df, traffic_source=traffic_source)
            self.texts = corpus
            embedding, _ = create_X_embedding_ft(corpus, self.model)
            return embedding
      
      def get_text(self) -> List[str]:
            if not self.texts:
                  return []

            if isinstance(self.texts[0], list):
                  return [" ".join([str(token) for token in tokens]) for tokens in self.texts]

            return self.texts

      @property
      def vector_size(self) -> int:
            return self.model.vector_size


class EmbeddingService:
      _instance: BaseEmbedder = None

      def __init__(cls):
            cls.text = None
            cls.embedding = None

      @classmethod
      def return_instance(cls):
            return cls._instance
      
      @classmethod
      def clear_instance(cls):
            cls._instance = None

      @classmethod
      def get_instance(cls, config_type: str, path_or_name: str, traffic_source: str = None) -> BaseEmbedder:
            if cls._instance is not None:
                  return cls._instance

            if config_type == "transformers":
                  cls._instance = TransformerEmbedder(model_name=path_or_name)
            elif config_type == "fasttext":
                  cls._instance = FastTextEmbedder(model_path=path_or_name, traffic_source=traffic_source)
            else:
                  raise ValueError("Tipo inválido. Use 'transformer' ou 'fasttext'.")
            return cls._instance

      @classmethod
      def process_and_encode(cls, df: pd.DataFrame) -> Tuple[np.ndarray, List]:

            if cls._instance is None:
                  raise RuntimeError("Embedder não inicializado! Chame get_instance primeiro.")
      
            embeddings = cls._instance.encode(df)
            print("Finishing encoding")
            texts = cls._instance.get_text()
            
            return embeddings, texts
      