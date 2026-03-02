import ipaddress
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import List, Dict, Any
from asyncio.log import logger
from torch.utils.data import DataLoader

# Importe as suas classes de IA existentes
from app.mil_attention.attention_based import AttentionMIL, MILBagDatasetLogical
from app.repositories.model_repository import ModelRepository
from app.services.embedding_service import EmbeddingService
from app.services.request_service import RequestService
from app.config.settings import settings


class ModelService:

      def __init__(
        self, 
        traffic_source: str,
        emb_config: str = "fasttext",
        hidden_dim: int = 256
      ):
            self.model_path = f"{settings.REPOSITORY_PATH}/{traffic_source}/{emb_config}/attention_mil_bundle.pth"
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

      def soft_bag_label(self, labels, min_pos=1, alpha=1.0):
            labels = np.array(labels)
            p = labels.mean() 
            
            if labels.sum() < min_pos:
                  return 0.0
            
            return float(min(1.0, p * alpha))
      
      def attention_entropy(self, A, eps=1e-8):
            A = A.clamp(min=eps)
            entropy = -torch.sum(A * torch.log(A), dim=1)
            return entropy.mean()


      def train(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 1) -> Dict[str, Any]:

            df = data.copy()
            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})

            mapeamento_mil = {"bots": 1, "unsafe": 0}
            df["decision_mil"] = df["decision"].map(mapeamento_mil)

             # agrupamento por bloco de ip
            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            #3 bag id -> utilizando pelo MIL Dataset para agrupar
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]
            print(df.head())

            logger.info("Processando e codificando textos do DataFrame...")
            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)

            logger.info("Agrupando requisições em Bags por Endereço IP...")
            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "decision_mil": list,
                  "ip": list
            }).reset_index()

            # bags_df["bag_label"] = bags_df["decision_mil"].apply(lambda labels: 1.0 if 1 in labels else 0.0)

            # aplicando soft bag_label
            bags_df["bag_label"] = bags_df["decision_mil"].apply(
                  lambda labels: self.soft_bag_label(labels, min_pos=2, alpha=1.5)
            )

            dataset = MILBagDatasetLogical(bags_df)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            modelo = AttentionMIL(input_dim=self.in_features, hidden_dim=self.hidden_dim).to(self.device)
            optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            modelo.train()
            logger.info(f"Iniciando treinamento ({epochs} épocas) no {self.device}...")

            accumulation_steps = 16

            optimizer.zero_grad()
            for epoch in range(epochs):
                  loss_acumulada = 0.0
                  for i, (bag, label, _) in enumerate(dataloader):

                        bag = bag.to(self.device)
                        # label = label.to(self.device).unsqueeze(1).float()
                        label = label.to(self.device).view(-1, 1)
                        logits, A = modelo(bag)

                        with torch.no_grad():
                              prob = torch.sigmoid(logits)
                        
                        # raw_loss = criterion(logits, label)

                        raw_loss = criterion(logits.view(-1), label.view(-1))

                        # loss_att = self.attention_entropy(A)
                        loss_acumulada += raw_loss.item()
                        # raw_loss_final = raw_loss + 0.01 * loss_att
                        
                        raw_loss_final = raw_loss
                        loss = raw_loss_final / accumulation_steps
                        loss.backward()
                        
                        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                              optimizer.step()
                              optimizer.zero_grad()
                  
                  loss_media = loss_acumulada / len(dataloader)
                  logger.info(f"Época {epoch+1}/{epochs} - Train Loss: {loss_media:.4f}")

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            torch.save({
                  "model_state_dict": modelo.state_dict(),
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

                  return {"status": "success", "loss_final": loss_acumulada/len(dataloader)}
            
      def _extract_ip_stack(self, ip_string):
        try:
            ip = ipaddress.ip_address(ip_string)
            if ip.version == 4:
                return str(ipaddress.ip_network(f"{ip_string}/24", strict=False))
            elif ip.version == 6:
                return str(ipaddress.ip_network(f"{ip_string}/48", strict=False))
        except ValueError:
            return "ip_invalido"
        
      def predict(self, df: pd.DataFrame) -> pd.DataFrame:

            # -----------------------------
            # 1️⃣ Processar embeddings
            # -----------------------------
            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)

            # -----------------------------
            # 2️⃣ Preparar MIL bags
            # -----------------------------
            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
            df["decision_mil"] = df["decision"].map({"bots": 1, "unsafe": 0})

            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "ip": list
            }).reset_index()

            # -----------------------------
            # 3️⃣ Carregar modelo
            # -----------------------------
            # modelo = AttentionMIL(in_features=300, hidden_dim=256)
            modelo = AttentionMIL(input_dim=self.in_features, hidden_dim=self.hidden_dim).to(self.device)
            checkpoint = torch.load(self.model_path, weights_only=False)
            modelo.load_state_dict(checkpoint["model_state_dict"])
            modelo.eval().to(self.device)

            resultados_finais = []

            # -----------------------------
            # 4️⃣ Inferência MIL
            # -----------------------------
            with torch.no_grad():
                  for _, row in bags_df.iterrows():
                        ip_atual = row["bag_id"]
                        bag_tensor = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        logits, attention = modelo(bag_tensor)
                        # Bag-level probability
                        certeza_bag = torch.sigmoid(logits).item()

                        print("certeza bag: ", certeza_bag)
                        classe_predita = 1 if certeza_bag > 0.5 else 0

                        # Instance-level attention
                        pesos = attention.squeeze().cpu().numpy()
                        if pesos.ndim == 0:
                              pesos = [pesos.item()]
                        else:
                              pesos = pesos.tolist()

                        # Instance-level probability (bag-level x attention)
                        inst_prob = [certeza_bag * p for p in pesos]

                        # Mapear para todas as requisições do IP
                        linhas_do_ip = df[df["bag_id"] == ip_atual].copy()
                        n_instances = len(linhas_do_ip)
                        linhas_do_ip["attention_weight"] = [pesos[i] if i < len(pesos) else 0.0 for i in range(n_instances)]
                        linhas_do_ip["instance_prob"] = [inst_prob[i] if i < len(inst_prob) else 0.0 for i in range(n_instances)]
                        linhas_do_ip["certeza_bag"] = round(certeza_bag * 100, 2) # em porcentagem 
                        linhas_do_ip["pred"] = classe_predita

                        resultados_finais.append(linhas_do_ip)

            df_completo = pd.concat(resultados_finais, ignore_index=True)

            # -----------------------------
            # 5️⃣ Métricas MIL no nível da bag
            # -----------------------------
            bag_labels = df_completo.groupby("bag_id")["decision_mil"].max()
            bag_preds  = df_completo.groupby("bag_id")["pred"].max()
            bag_probs  = df_completo.groupby("bag_id")["certeza_bag"].max() / 100  # scale back to [0,1]

            acuracia_bag = (bag_labels == bag_preds).mean()
            total_erros_bag = (bag_labels != bag_preds).sum()
            fp_bag = ((bag_labels == 0) & (bag_preds == 1)).sum()
            fn_bag = ((bag_labels == 1) & (bag_preds == 0)).sum()

            # Métricas contínuas
            try:
                  roc_auc = roc_auc_score(bag_labels, bag_probs)
                  pr_auc  = average_precision_score(bag_labels, bag_probs)
            except:
                  roc_auc = pr_auc = None

            # Log das métricas
            print(f"✅ MIL Bag-level Metrics:")
            print(f"Total bags: {len(bag_labels)}")
            print(f"Accuracy (bag-level): {acuracia_bag:.4f}")
            print(f"Total bag prediction errors: {total_erros_bag}")
            print(f"False Positives: {fp_bag}")
            print(f"False Negatives: {fn_bag}")
            print(f"ROC-AUC: {roc_auc}")
            print(f"PR-AUC: {pr_auc}")

            # Retorna dataframe completo (instance-level + bag-level)
            return df_completo
      
      def predict_with_instance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
    
            # -----------------------------
            # Processar embeddings
            # -----------------------------
            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)

            # Preparar MIL bags
            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
            df["decision_mil"] = df["decision"].map({"bots": 1, "unsafe": 0})
            df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
            df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
            df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

            bags_df = df.groupby("bag_id").agg({
                  "embedding": list,
                  "ip": list
            }).reset_index()

            # -----------------------------
            # Carregar modelo
            # -----------------------------
            modelo = AttentionMIL(input_dim=self.in_features, hidden_dim=self.hidden_dim).to(self.device)
            checkpoint = torch.load(self.model_path, weights_only=False)
            modelo.load_state_dict(checkpoint["model_state_dict"])
            modelo.eval().to(self.device)

            resultados_finais = []

            # -----------------------------
            # Inferência MIL
            # -----------------------------
            with torch.no_grad():
                  for _, row in bags_df.iterrows():
                        bag_tensor = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(self.device)
                        logits, attention = modelo(bag_tensor)
                        
                        certeza_bag = torch.sigmoid(logits).item()
                        pred_bag = 1 if certeza_bag > 0.5 else 0

                        # Instance-level
                        a = attention.squeeze().cpu().numpy()
                        if a.ndim == 0: a = [a.item()]
                        instance_prob = [certeza_bag * p for p in a]
                        entropy = -np.sum([p*np.log(p+1e-8) for p in a])  # atenção normalizada

                        # Mapear para todas as requisições do IP
                        linhas = df[df["bag_id"] == row["bag_id"]].copy()
                        n_instances = len(linhas)
                        linhas["attention_weight"] = [a[i] if i < len(a) else 0.0 for i in range(n_instances)]
                        linhas["instance_prob"] = [instance_prob[i] if i < len(instance_prob) else 0.0 for i in range(n_instances)]
                        linhas["bag_prob"] = certeza_bag
                        linhas["bag_pred"] = pred_bag
                        linhas["attention_entropy"] = entropy

                        resultados_finais.append(linhas)

            df_completo = pd.concat(resultados_finais, ignore_index=True)

            # -----------------------------
            # Bag-level metrics
            # -----------------------------
            bag_labels = df_completo.groupby("bag_id")["decision_mil"].max()
            bag_preds  = df_completo.groupby("bag_id")["bag_pred"].max()
            bag_probs  = df_completo.groupby("bag_id")["bag_prob"].max()

            acuracia_bag = (bag_labels == bag_preds).mean()
            total_erros_bag = (bag_labels != bag_preds).sum()
            fp_bag = ((bag_labels == 0) & (bag_preds == 1)).sum()
            fn_bag = ((bag_labels == 1) & (bag_preds == 0)).sum()

            roc_auc = roc_auc_score(bag_labels, bag_probs)
            pr_auc  = average_precision_score(bag_labels, bag_probs)

            print(f"✅ MIL Bag-level Metrics:")
            print(f"Total bags: {len(bag_labels)}")
            print(f"Accuracy (bag-level): {acuracia_bag:.4f}")
            print(f"Total bag prediction errors: {total_erros_bag}")
            print(f"False Positives: {fp_bag}")
            print(f"False Negatives: {fn_bag}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")

            # -----------------------------
            # Instance-level summary
            # -----------------------------
            print("🧩 Instance-level summary:")
            print(f"Avg attention entropy: {df_completo['attention_entropy'].mean():.4f}")
            print(f"Avg instance probability: {df_completo['instance_prob'].mean():.4f}")
            print(f"Min instance probability: {df_completo['instance_prob'].min():.4f}")
            print(f"Max instance probability: {df_completo['instance_prob'].max():.4f}")

            return df_completo

      # def predict(self, data: pd.DataFrame) -> pd.DataFrame:
      #       """
      #       Runs Attention MIL inference in a way that is consistent
      #       with the batch evaluation pipeline.

      #       - Prediction is performed at BAG level
      #       - Ground truth is kept explicit and binary
      #       - Line-level output is only a projection of bag prediction
      #       """

      #       df = data.copy()

      #       # --------------------------------------------------
      #       # 1. Explicit and consistent ground truth
      #       # --------------------------------------------------
      #       if "decision_mil" not in df.columns:
      #             df["decision_mil"] = (
      #                   df["decision"]
      #                   .str.lower()
      #                   .map({"bots": 1, "bot": 1, "unsafe": 0})
      #             )

      #       # --------------------------------------------------
      #       # 2. Embedding (same path used in training)
      #       # --------------------------------------------------
      #       embeddings_matrix, _ = self.emb_service.process_and_encode(df)
      #       df["embedding"] = list(embeddings_matrix)

      #       # --------------------------------------------------
      #       # 3. Bag construction (MUST match training logic)
      #       # --------------------------------------------------
      #       df["ip_block"] = df["ip"].apply(self._extract_ip_stack)
      #       df["ip_api_isp"] = df["ip_api_isp"].fillna("isp_unknow")
      #       df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

      #       bags_df = (
      #             df.groupby("bag_id")
      #             .agg({
      #                   "embedding": list,
      #                   "decision_mil": "max",  # bag-level ground truth
      #                   "ip": list
      #             })
      #             .reset_index()
      #       )

      #       print("BAGS FORMED (CONSISTENT WITH TRAINING)")
      #       print(bags_df.head())

      #       # --------------------------------------------------
      #       # 4. Model loading (identical to batch pipeline)
      #       # --------------------------------------------------
      #       model = AttentionMIL(
      #             in_features=self.in_features,
      #             hidden_dim=self.hidden_dim
      #       ).to(self.device)

      #       checkpoint = torch.load(
      #             self.model_path,
      #             map_location=self.device
      #       )

      #       model.load_state_dict(checkpoint["model_state_dict"])
      #       model.eval()

      #       # --------------------------------------------------
      #       # 5. Inference (PURE bag-level)
      #       # --------------------------------------------------
      #       bag_predictions = []

      #       with torch.no_grad():
      #             for _, row in bags_df.iterrows():
      #                   bag_tensor = (
      #                   torch.tensor(row["embedding"], dtype=torch.float32)
      #                   .unsqueeze(0)
      #                   .to(self.device)
      #                   )

      #                   pred, attention = model(bag_tensor)

      #                   prob = pred.item()
      #                   bag_class = 1 if prob > 0.5 else 0

      #                   weights = attention.squeeze().cpu().numpy()
      #                   if weights.ndim == 0:
      #                         weights = [weights.item()]
      #                   else:
      #                         weights = weights.tolist()

      #                   bag_predictions.append({
      #                         "bag_id": row["bag_id"],
      #                         "pred": bag_class,
      #                         "bag_probability": round(prob * 100, 2),
      #                         "decision_mil": row["decision_mil"],
      #                         "attention_weights": weights
      #                   })

      #       df_bags_pred = pd.DataFrame(bag_predictions)

      #       # --------------------------------------------------
      #       # 6. Project bag prediction back to rows (optional)
      #       # --------------------------------------------------
      #       df_out = df.merge(
      #             df_bags_pred[["bag_id", "pred", "bag_probability"]],
      #             on="bag_id",
      #             how="left"
      #       )

      #       # --------------------------------------------------
      #       # 7. Cleanup
      #       # --------------------------------------------------
      #       if "embedding" in df_out.columns:
      #             df_out = df_out.drop(columns=["embedding"])

      #       return df_out
      
      