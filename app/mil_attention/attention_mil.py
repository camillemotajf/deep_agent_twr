from asyncio.log import logger
import ipaddress
import os

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.config.settings import settings
from app.repositories.model_repository import ModelRepository
from app.services.embedding_service import EmbeddingService

class MILAttentionLayer(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super(MILAttentionLayer, self).__init__()

            self.input_dim = input_dim
            self.weight_params_dim = weight_params_dim
            self.used_gated = use_gated

            # v: projeção inicial
            self.v_weight_params = nn.Linear(input_dim, weight_params_dim, bias=False)

            # w: projeção final para score
            self.w_weight_params = nn.Linear(weight_params_dim, 1, bias=False)

            # u: mecanismo de gating
            if self.use_gated:
                  self.u_weight_params = nn.Linear(input_dim, weight_params_dim, bias=False)
            else:
                  self.register_parameter("u_weight_params", None)

      def forward(self, inputs):

            if isinstance(inputs, list):
                  h = torch.stack(inputs)
            else:
                  h = inputs

            a_v = torch.tanh(self.v_weight_params(h))

            if self.used_gated:
                  a_u = torch.sigmoid(self.u_weight_params(h))

                  a_v = a_v * a_u

            attention_scores = self.w_weight_params(a_v)
            alpha = F.softmax(attention_scores, dim=0)

            return [alpha[i] for i in range(alpha.shape[0])]

class MILBotClassifier(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super().__init__()
            
            # 1. A sua Camada de Atenção
            self.attention = MILAttentionLayer(input_dim, weight_params_dim)
            
            # 2. O Classificador Final (Pode ajustar as camadas ocultas)
            self.classifier = nn.Sequential(
                  nn.Linear(input_dim, 64),
                  nn.ReLU(),
                  nn.Dropout(0.3),
                  nn.Linear(64, 1) # Retorna "logits" (sem sigmoid, melhor para estabilidade no treino)
            )

      def forward(self, x):
            """
            Args:
            x: Tensor de formato (batch_size, bag_size, input_dim)
            """
            # Preparar para a camada de atenção (que espera uma lista)
            # Transpõe para (bag_size, batch_size, input_dim)
            x_swapped = torch.swapaxes(x, 0, 1)
            x_list = list(torch.unbind(x_swapped, dim=0))
            
            # Obter os scores de atenção: Lista de tamanho bag_size de tensores (batch_size, 1)
            attention_scores = self.attention(x_list)
            
            # Empilhar os scores para multiplicar: (bag_size, batch_size, 1)
            att_stacked = torch.stack(attention_scores)
            
            # Multiplicar as instâncias originais pelos scores de atenção
            # O PyTorch aplica broadcasting na dimensão 1
            weighted_instances = x_swapped * att_stacked
            
            # Somar todas as instâncias da bag (dim=0) para obter a representação única da bag
            # Formato final: (batch_size, input_dim)
            bag_representation = torch.sum(weighted_instances, dim=0)
            
            # Passar pelo classificador final
            logits = self.classifier(bag_representation)
            
            return logits


class MILBagProcessor:
    """
    Processa um DataFrame para gerar Bags para Multiple Instance Learning (MIL).
    
    Args:
        bag_size (int): Tamanho fixo de cada bag (número de instâncias).
        positive_class (int): Rótulo que define a classe positiva (ex: 1 para 'bots').
    """
    def __init__(self, bag_size: int, positive_class: int = 1):
        self.bag_size = bag_size
        self.positive_class = positive_class

    def transform(self, df: pd.DataFrame):

        bags_df = df.groupby("bag_id").agg({
            "embedding": list,
            "decision_mil": list,
            "decision": list,
            "ip": list
        }).reset_index()

        bags_data = []
        bag_labels = []
        positive_bags_count = 0

        for _, row in bags_df.iterrows():
            instance_embeddings = row["embedding"]
            instance_labels = row["decision"]
            num_instances = len(instance_embeddings)
            
            # Ajustar o tamanho da bag para o 'self.bag_size' fixo
            if num_instances >= self.bag_size:
                # Subamostragem sem reposição (pega num subconjunto aleatório)
                indices = np.random.choice(num_instances, self.bag_size, replace=False)
            else:
                # Sobreamostragem com reposição (repete instâncias para preencher a bag)
                indices = np.random.choice(num_instances, self.bag_size, replace=True)
            
            sampled_embeddings = [instance_embeddings[i] for i in indices]
            sampled_labels = [instance_labels[i] for i in indices]
            
            # 3. Definir o rótulo da Bag (A regra de ouro do MIL)
            # Se houver PELO MENOS UMA instância positiva na bag, a bag é positiva.
            if self.positive_class in sampled_labels:
                bag_label = 1
                positive_bags_count += 1
            else:
                bag_label = 0
                
            bags_data.append(sampled_embeddings)
            bag_labels.append([bag_label])

        print(f"Total de Bags geradas: {len(bags_df)}")
        print(f"  -> Bags Positivas (com bots): {positive_bags_count}")
        print(f"  -> Bags Negativas (seguras): {len(bags_df) - positive_bags_count}")
        bags_tensor = torch.tensor(np.array(bags_data), dtype=torch.float32)
        labels_tensor = torch.tensor(bag_labels, dtype=torch.float32)
        
        bags_swapped = torch.swapaxes(bags_tensor, 0, 1)
      
        bags_list_of_tensors = list(torch.unbind(bags_swapped, dim=0))

        return bags_tensor, labels_tensor
    

class MILAttetionService:

      def __init__(
        self, 
        traffic_source: str,
        emb_config: str = "fasttext",
        hidden_dim: int = 256,
        model_path: str | None = None
      ):
            if not model_path:
                  self.model_path = f"{settings.REPOSITORY_PATH}/{traffic_source}/{emb_config}/attention_mil_bundle.pth"
            else: 
                 self.model_path = f"{model_path}/attention_mil_teste.pth"
            self.hidden_dim = hidden_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.traffic_source = traffic_source

            self.repo_s3 = ModelRepository()
            self.emb_config = emb_config
            self.emb_service = EmbeddingService()

            if self.emb_config == "fasttext":
                  base_emb_path = f"{settings.REPOSITORY_PATH}/{traffic_source}/fasttext_{traffic_source}.model"

                  self.repo_s3.sync_model(
                        local_path=base_emb_path, 
                        is_base_embedding=True
                  )
                  self.emb_service_instance = self.emb_service.get_instance(config_type=emb_config, path_or_name=base_emb_path, repo_s3=self.repo_s3)

            elif emb_config == "transformers":
                  self.emb_service_instance = self.emb_service.get_instance(config_type=self.emb_config, path_or_name=settings.TRANSFORMER_MODEL)

            self.in_features = self.emb_service_instance.vector_size

            self.repo_s3.sync_model(
                  local_path=self.model_path, 
                  is_base_embedding=False,
                  traffic_source=traffic_source,
                  emb_config=self.emb_config
            )

      def _extract_ip_stack(self, ip_string):
            try:
                  ip = ipaddress.ip_address(ip_string)
                  if ip.version == 4:
                        return str(ipaddress.ip_network(f"{ip_string}/24", strict=False))
                  elif ip.version == 6:
                        return str(ipaddress.ip_network(f"{ip_string}/48", strict=False))
            except ValueError:
                  return "ip_invalido"

      def _soft_bag_label(self, labels, min_pos=1, alpha=1.0):
            labels = np.array(labels)
            p = labels.mean() 
            
            if labels.sum() < min_pos:
                  return 0.0
            
            return float(min(1.0, p * alpha))
      
      def _attention_entropy(self, A, eps=1e-8):
            A = A.clamp(min=eps)
            entropy = -torch.sum(A * torch.log(A), dim=1)
            return entropy.mean()

      def _extract_ip_stack(self, ip_string):
        try:
            ip = ipaddress.ip_address(ip_string)
            if ip.version == 4:
                return str(ipaddress.ip_network(f"{ip_string}/24", strict=False))
            elif ip.version == 6:
                return str(ipaddress.ip_network(f"{ip_string}/48", strict=False))
        except ValueError:
            return "ip_invalido"

      def train(self, data: pd.DataFrame, epochs: int = 10):

            df = data.copy()
            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
            mapeamento_mil = {"bots": 1, "unsafe": 0}
            df["decision_mil"] = df["decision"].map(mapeamento_mil)

             # agrupamento por bloco de ip
            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

            logger.info("Processando e codificando textos do DataFrame...")
            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)

            logger.info("Agrupando requisições em Bags por Endereço IP...")
            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "decision_mil": list,
                  "ip": list
            }).reset_index()

            # aplicando soft bag_label
            bags_df["bag_label"] = bags_df["decision_mil"].apply(
                  lambda labels: self._soft_bag_label(labels, min_pos=2, alpha=1.5)
            )

            processor = MILBagProcessor(bag_size=3, positive_class="bots")
            bags_tensor, labels_tensor = processor.transform(df)
            dataset = TensorDataset(bags_tensor, labels_tensor)
            BATCH_SIZE = 32
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            model = MILBotClassifier(input_dim=self.in_features, weight_params_dim=128)

            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            logger.info(f"Iniciando treinamento ({epochs} épocas) no {self.device}...")

            for epoch in range(epochs):
                  model.train()
                  epoch_loss = 0.0
                  correct_preds = 0.0
                  total_samples = 0 

                  progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

                  for batch_x, batch_y in progress_bar:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        optimizer.zero_grad()
                        logits = model(batch_x)

                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        preds = torch.sigmoid(logits).round()
                        correct_preds += (preds == batch_y).sum().item()
                        total_samples += batch_y.size(0)

                        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct_preds/total_samples})

                  print(f"End of epoch {epoch+1} | Mean Loss: {epoch_loss/len(train_loader):.4f}")

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)


            print("Salving model at: ", self.model_path)
            torch.save({
                  "model_state_dict": model.state_dict(),
                  "config": {"in_features": self.in_features, "hidden_dim": self.hidden_dim},
            }, self.model_path)
