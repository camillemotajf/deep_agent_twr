from asyncio.log import logger
import ipaddress
import os

import torch
from torch import optim
from torch import random
from torch.amp import autocast, GradScaler
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from app.config.settings import settings
from app.mil_attention.attention_mil import MILBagProcessor, MILBotClassifier
from app.repositories.model_repository import ModelRepository
from app.services.embedding_service import EmbeddingService


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

      def train_bucketing(self, data: pd.DataFrame, epochs: int = 10):

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
            train_loader = processor.bucketing(
                  bags_list=bags_tensor,
                  labels_list=labels_tensor
            )
            # dataset = MILDataset(bags_tensor, labels_tensor)
            # train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

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

                  batches = list(train_loader)
                  np.random.shuffle(batches)

                  progress_bar = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}")
                  optimizer.zero_grad()

                  for batch_x, batch_y, mask in progress_bar:

                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device).view(-1, 1)
                        mask = mask.to(self.device)

                        with autocast(device_type="cuda"):
                              logits = model(batch_x, mask=mask)
                              loss = criterion(logits, batch_y)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        epoch_loss += loss.item()

                        # epoch_loss += loss.item() * batch
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

            deploy = self.repo_s3.deploy_model(
                 local_model_path=self.model_path,
                 is_base_mebedding=False,
                 traffic_source=self.traffic_source,
                 emb_config=self.emb_config
            )

            if deploy:
                  logger.info("Classification Model saved and versioning in s3")

      def train_bags_accumulated(self, data: pd.DataFrame, epochs=30, accumulation_steps=32):
            """
            Treina o modelo MIL avaliando uma bag exata por vez (sem padding/bucketing),
            acumulando os gradientes para estabilidade e usando balanceamento de classe.
            """

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
            bags_list, labels_list = processor.transform(df)
            
            logger.info("Agrupando requisições em Bags por Endereço IP...")


            # 2. SETUP DO MODELO
            model = MILBotClassifier(input_dim=self.in_features, weight_params_dim=128, use_gated=True).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            scaler = GradScaler(device=self.device) 

            logger.info(f"Iniciando treinamento ({epochs} épocas) no {self.device} com Accumulation={accumulation_steps}...")

            for epoch in range(epochs):
                  model.train()
                  epoch_loss = 0.0
                  correct_preds = 0
                  total_samples = len(bags_list)
                  
                  # Junta e embaralha os dados a cada época
                  data_train = list(zip(bags_list, labels_list))
                  np.random.shuffle(data_train)
                  
                  progress_bar = tqdm(data_train, desc=f"Epoch {epoch+1}/{epochs}")
                  
                  optimizer.zero_grad()
                  
                  for idx, (bag, label) in enumerate(progress_bar):
                        
                        # bag tem formato [N, D]. O modelo espera [Batch, N, D]. 
                        # O unsqueeze(0) cria um batch falso de tamanho 1: [1, N, D]
                        bag_x = bag.unsqueeze(0).to(self.device)
                        bag_y = label.unsqueeze(0).to(self.device) # [1, 1]
                        
                        # Mixed Precision para não estourar a VRAM da GPU
                        with autocast(device_type="cuda"):
                              # MÁSCARA = None (Não precisamos dela, pois não há padding!)
                              logits = model(bag_x, mask=None)
                              
                              # Calcula o Loss e DIVIDE pelos steps de acumulação
                              loss = criterion(logits, bag_y)
                              loss = loss / accumulation_steps
                              
                              # Calcula os gradientes (eles vão se somando na memória)
                              scaler.scale(loss).backward()
                              
                              # Apenas atualiza os pesos quando atingir o número de steps 
                              # ou quando for a última bag da época
                              if (idx + 1) % accumulation_steps == 0 or (idx + 1) == total_samples:
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad() # Limpa a memória de gradientes para o próximo ciclo
                              
                              # Métricas para acompanhamento visual
                              epoch_loss += (loss.item() * accumulation_steps) # Multiplica para exibir o loss real
                        
                        with torch.no_grad():
                              pred = torch.sigmoid(logits).round()
                              correct_preds += int(pred.item() == bag_y.item())
                        
                        progress_bar.set_postfix({
                              'loss': f"{(epoch_loss / (idx + 1)):.4f}", 
                              'acc': f"{(correct_preds / (idx + 1)):.4f}"
                        })

                  print(f"End of epoch {epoch+1} | Mean Loss: {epoch_loss/len(bags_list):.4f}")

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            print("Salving model at: ", self.model_path)
            torch.save({
                  "model_state_dict": model.state_dict(),
                  "config": {"in_features": self.in_features, "hidden_dim": self.hidden_dim},
            }, self.model_path)

            deploy = self.repo_s3.deploy_model(
                 local_model_path=self.model_path,
                 is_base_mebedding=False,
                 traffic_source=self.traffic_source,
                 emb_config=self.emb_config
            )

            if deploy:
                  logger.info("Classification Model saved and versioning in s3")
                        
            return model

      
            

      def predict(self, df: pd.DataFrame, model_loaded=None) -> pd.DataFrame:
            
            embedding_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embedding_matrix)

            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
            df["decision_mil"] = df["decision"].map({"bots": 1, "unsafe": 0})

            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "decision_mil": list,
                  "headers": list,
                  "request": list,
                  "ip": list
            }).reset_index()

            if model_loaded:
                  model = model_loaded 
            else:
                  model = MILBotClassifier(input_dim=self.in_features, weight_params_dim=128, use_gated=True)
                  cp = torch.load(self.model_path, weights_only=False)
                  model.load_state_dict(cp["model_state_dict"])

            model.eval().to(self.device)

            all_probs = []
            all_attentions = []

            with torch.no_grad():
                  for _, row in bags_df.iterrows():
                        bag_x = torch.tensor(np.array(row["embedding"]), dtype=torch.float32)
                        batch_x = bag_x.unsqueeze(0).to(self.device)
                        logits, attention_w = model(batch_x, return_attention=True)
                        
                        probs = torch.sigmoid(logits).squeeze(-1)
                        if probs.dim() == 0:
                              probs = probs.unsqueeze(0)
                        
                        all_probs.extend(probs.cpu().numpy())
                        all_attentions.append(list(attention_w.cpu().numpy().flatten()))


            bags_df["mil_bot_probability"] = np.array(all_probs)
            bags_df["mil_prediction"] = (np.array(all_probs) >= 0.5).astype(int)
            bags_df["attention_weight"] = all_attentions

            instances_df = bags_df.explode([
                 "ip", 
                 "embedding", 
                 "headers",
                 "request",
                 "attention_weight",
                 "decision_mil"
            ]).reset_index(drop=True)

            instances_df["decision_mil"] = instances_df["decision_mil"].astype(int)
            instances_df["attention_weight"] = instances_df["attention_weight"].astype(float)

            return instances_df