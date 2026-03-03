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

class MILAttentionLayer(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super(MILAttentionLayer, self).__init__()

            self.input_dim = input_dim
            self.weight_params_dim = weight_params_dim
            self.use_gated = use_gated

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

            if self.use_gated:
                  a_u = torch.sigmoid(self.u_weight_params(h))

                  a_v = a_v * a_u

            attention_scores = self.w_weight_params(a_v)
            alpha = F.softmax(attention_scores, dim=0)

            return [alpha[i] for i in range(alpha.shape[0])]

class MILBotClassifier(nn.Module):
      def __init__(self, input_dim, weight_params_dim, use_gated=True):
            super().__init__()
            
            # 1. A sua Camada de Atenção
            self.attention = MILAttentionLayer(input_dim, weight_params_dim, use_gated=use_gated)
            
            # 2. O Classificador Final (Pode ajustar as camadas ocultas)
            self.classifier = nn.Sequential(
                  nn.Linear(input_dim, 64),
                  nn.ReLU(),
                  nn.Dropout(0.3),
                  nn.Linear(64, 1) # Retorna "logits" (sem sigmoid, melhor para estabilidade no treino)
            )

      def forward(self, x, return_attention=False):
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

            if return_attention:
                  # Transpomos para (batch_size, bag_size) para ficar fácil de ler
                  weights = att_stacked.permute(1, 0, 2).squeeze(-1)
                  return logits, weights
            
            return logits


class MILBagProcessor:
      """
      Processa um DataFrame para gerar Bags para Multiple Instance Learning (MIL).
      Gera bags de tamanhos variáveis (tamanho real do histórico) para uso com batch_size=1.
      """
      def __init__(self, positive_class="bots"):
            # Não precisamos mais do bag_size!
            self.positive_class = positive_class

      def transform(self, df: pd.DataFrame):
            # Agrupa as instâncias pelo bag_id
            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "decision": list,
                  "ip": list
            }).reset_index()

            bags_list = []
            labels_list = []
            positive_bags_count = 0

            for _, row in bags_df.iterrows():
                  instance_embeddings = row["embedding"]
                  instance_labels = row["decision"]
                  
                  # 1. Cria o Tensor da Bag com o seu tamanho original [N_instancias, features]
                  bag_tensor = torch.tensor(np.array(instance_embeddings), dtype=torch.float32)
                  
                  # 2. Define o rótulo da Bag (se há pelo menos 1 bot, a bag é bot)
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

            # Retorna as listas de tensores
            return bags_list, labels_list


# ==========================================================
# DATASET CUSTOMIZADO PARA TAMANHOS VARIÁVEIS
# ==========================================================
class MILDataset(Dataset):
      def __init__(self, bags_list, labels_list):
            self.bags = bags_list
            self.labels = labels_list

      def __len__(self):
            return len(self.labels)

      def __getitem__(self, idx):
            # Retorna a bag e o label daquela posição
            return self.bags[idx], self.labels[idx]
    

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
            
            processor = MILBagProcessor(positive_class="bots")
            bags_tensor, labels_tensor = processor.transform(df)
            dataset = MILDataset(bags_tensor, labels_tensor)
            train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

            model = MILBotClassifier(input_dim=self.in_features, weight_params_dim=128).to(self.device)

            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            logger.info(f"Iniciando treinamento ({epochs} épocas) no {self.device}...")
            scaler = GradScaler(device=self.device)

            for epoch in range(epochs):
                  model.train()
                  epoch_loss = 0.0
                  correct_preds = 0.0
                  total_samples = 0 

                  progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

                  batch = 32
                  optimizer.zero_grad()

                  for i, (batch_x, batch_y) in enumerate(progress_bar):
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        # optimizer.zero_grad()
                        with autocast(device_type="cuda"):
                              logits = model(batch_x)
                              loss = criterion(logits, batch_y)
                              loss = loss / batch

                        scaler.scale(loss).backward()

                        if (i + 1) % batch == 0 or (i + 1) == len(train_loader):
                              scaler.step(optimizer)
                              scaler.update()
                              optimizer.zero_grad()

                        epoch_loss += loss.item() * batch
                        batch_y = batch_y.float()

                        with torch.no_grad():
                              preds = torch.sigmoid(logits).squeeze().round()
                              correct_preds += (preds == batch_y.squeeze()).sum().item()
                              total_samples += batch_y.size(0)

                        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct_preds/total_samples})

                  print(f"End of epoch {epoch+1} | Mean Loss: {epoch_loss/len(train_loader):.4f}")

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)


            print("Salving model at: ", self.model_path)
            torch.save({
                  "model_state_dict": model.state_dict(),
                  "config": {"in_features": self.in_features, "hidden_dim": self.hidden_dim},
            }, self.model_path)



      def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            
            # 1. Prepara os dados iniciais
            embedding_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embedding_matrix)

            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
            df["decision_mil"] = df["decision"].map({"bots": 1, "unsafe": 0})

            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

            # Agrupa as instâncias (aqui as bags ficam com tamanhos originais: 1, 5, 10, etc)
            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "decision_mil": list,
                  "ip": list
            }).reset_index()

            # 2. Carrega o Modelo
            model = MILBotClassifier(input_dim=self.in_features, weight_params_dim=128, use_gated=True)
            cp = torch.load(self.model_path, weights_only=False)
            model.load_state_dict(cp["model_state_dict"])
            model.eval().to(self.device)

            all_probs = []
            all_attentions = []

            # 3. Inferência (Sem MILBagProcessor e Sem Lote)
            # Passamos o histórico inteiro de cada IP de uma vez só
            with torch.no_grad():
                  for _, row in bags_df.iterrows():
                        # Converte as requisições do IP atual num Tensor
                        bag_x = torch.tensor(np.array(row["embedding"]), dtype=torch.float32)
                        
                        # Adiciona a dimensão de batch artificial [1, N_requisicoes, features]
                        batch_x = bag_x.unsqueeze(0).to(self.device)
                        
                        # Modelo avalia o histórico completo sem cortes
                        logits, attention_w = model(batch_x, return_attention=True)
                        
                        probs = torch.sigmoid(logits).squeeze(-1)
                        if probs.dim() == 0:
                              probs = probs.unsqueeze(0)
                        
                        all_probs.extend(probs.cpu().numpy())
                        
                        # Achata a atenção [1, N] para uma lista de tamanho N
                        all_attentions.append(list(attention_w.cpu().numpy().flatten()))

            # 4. Acoplando os resultados de volta
            # Como processamos 1 por 1, os tamanhos são RIGOROSAMENTE iguais
            bags_df["mil_bot_probability"] = np.array(all_probs)
            bags_df["mil_prediction"] = (np.array(all_probs) >= 0.5).astype(int)
            bags_df["attention_weight"] = all_attentions

            # 5. Explodindo de volta para nível de Instância (sem precisar truncar)
            instances_df = bags_df.explode([
                 "ip", 
                 "embedding", 
                 "attention_weight",
                 "decision_mil"
            ]).reset_index(drop=True)
            
            # Ordenamos para ver rapidamente quem foi o maior culpado
            # instances_df = instances_df.sort_values(by=["bag_id", "attention_weight"], ascending=[True, False])
            instances_df["decision_mil"] = instances_df["decision_mil"].astype(int)
            instances_df["attention_weight"] = instances_df["attention_weight"].astype(float)

            return instances_df


