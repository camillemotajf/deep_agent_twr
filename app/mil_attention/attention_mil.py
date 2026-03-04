from asyncio.log import logger
import ipaddress
import os

import torch
from torch import optim
# from torch.cuda.amp import GradScaler
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.config.settings import settings
from app.repositories.model_repository import ModelRepository
from app.services.embedding_service import EmbeddingService

from torch.nn.utils.rnn import pad_sequence

class MILAttentionLayer(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super(MILAttentionLayer, self).__init__()

            self.input_dim = input_dim
            self.weight_params_dim = weight_params_dim
            self.use_gated = use_gated

            self.v_weight_params = nn.Linear(input_dim, weight_params_dim, bias=False)

            self.w_weight_params = nn.Linear(weight_params_dim, 1, bias=False)

            if self.use_gated:
                  self.u_weight_params = nn.Linear(input_dim, weight_params_dim, bias=False)
            else:
                  self.register_parameter("u_weight_params", None)

      def forward(self, inputs, mask=None):

            if isinstance(inputs, list):
                  h = torch.stack(inputs)
            else:
                  h = inputs

            a_v = torch.tanh(self.v_weight_params(h))

            if self.use_gated:
                  a_u = torch.sigmoid(self.u_weight_params(h))
                  a_v = a_v * a_u

            attention_scores = self.w_weight_params(a_v)

            if mask is not None:
                  attention_scores = attention_scores.masked_fill(~mask, float('-inf'))


            alpha = F.softmax(attention_scores, dim=1)

            return alpha

class MILBotClassifier(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super().__init__()
            
            self.attention = MILAttentionLayer(input_dim, weight_params_dim, use_gated=use_gated)
            
            self.classifier = nn.Sequential(
                  nn.Linear(input_dim, 64),
                  nn.ReLU(),
                  nn.Dropout(0.3),
                  nn.Linear(64, 1) 
            )

      def forward(self, x, mask=None, return_attention=False):
         
            attention_scores = self.attention(x, mask)  # [B, N, 1]

            weighted_instances = x * attention_scores  # [B, N, D]

            bag_representation = torch.sum(weighted_instances, dim=1)  # [B, D]

            logits = self.classifier(bag_representation)

            if return_attention:
                  weights = attention_scores.squeeze(-1)  # [B, N]
                  return logits, weights

            return logits

class MILBagProcessor:
      
      def __init__(self, positive_class="bots"):
            self.positive_class = positive_class


      def mil_collate_fn(self, batch):
           
            bags = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            lengths = torch.tensor([len(bag) for bag in bags])

            padded_bags = pad_sequence(bags, batch_first=True, padding_value=0.0)

            batch_size, max_len, _ = padded_bags.shape
            mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1) 
            labels = torch.stack(labels)
            
            return padded_bags, labels, mask
      
      def bucketing(self, bags_list, labels_list, batch_size=32):
            data_with_lengths = [(bag, label, len(bag)) for bag, label in zip(bags_list, labels_list)]
            data_with_lengths.sort(key=lambda x: x[2])

            sorted_bags = [x[0] for x in data_with_lengths]
            sorted_labels = [x[1] for x in data_with_lengths]

            dataset = MILDataset(sorted_bags, sorted_labels)

            train_loader = DataLoader(
                  dataset=dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  collate_fn=self.mil_collate_fn,
                  pin_memory=True
            )

            return train_loader


      def transform(self, df: pd.DataFrame):
            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "headers": list,
                  "request": list,
                  "decision": list,
                  "ip": list
            }).reset_index()

            bags_list = []
            labels_list = []
            positive_bags_count = 0

            for _, row in bags_df.iterrows():
                  instance_embeddings = row["embedding"]
                  instance_labels = row["decision"]
                  
                  bag_tensor = torch.tensor(np.array(instance_embeddings), dtype=torch.float32)
                  
                  if self.positive_class in instance_labels:
                        bag_label = 1.0
                        positive_bags_count += 1
                  else:
                        bag_label = 0.0
                        
                  bags_list.append(bag_tensor)
                  labels_list.append(torch.tensor([bag_label], dtype=torch.float32))

            print(f"Total de Bags Processadas: {len(bags_list)}")
            print(f"  -> Bags Positivas (com bots): {positive_bags_count}")
            print(f"  -> Bags Negativas (seguras): {len(bags_list) - positive_bags_count}")

            return bags_list, labels_list


class MILDataset(Dataset):
      def __init__(self, bags_list, labels_list):
            self.bags = bags_list
            self.labels = labels_list

      def __len__(self):
            return len(self.labels)

      def __getitem__(self, idx):
            # Retorna a bag e o label daquela posição
            return self.bags[idx], self.labels[idx]
    


