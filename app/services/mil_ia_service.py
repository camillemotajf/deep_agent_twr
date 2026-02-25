import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import List, Dict, Any
from asyncio.log import logger
from torch.utils.data import DataLoader

# Importe as suas classes de IA existentes
from app.mil_attention.attention_based import AttentionMIL, MILBagDatasetLogical
from app.services.embedding_service import EmbeddingService
from app.services.request_service import RequestService

REPOSITORY_PATH = "G:/Meu Drive/TWR/data"
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

class ModelService:

      def __init__(
        self, 
        traffic_source: str,
        emb_config: str = "fasttext",
        hidden_dim: int = 256
      ):
            self.model_path = f"{REPOSITORY_PATH}/{traffic_source}/{emb_config}/attention_mil_bundle.pth"
            self.hidden_dim = hidden_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if emb_config == "fasttext":
                  self.emb_config = EmbeddingService.get_instance(config_type=emb_config, path_or_name=f"{REPOSITORY_PATH}/{traffic_source}/fasttext_{traffic_source}.model")

            elif emb_config == "transformers":
                  self.emb_config = EmbeddingService.get_instance(config_type=emb_config, path_or_name=TRANSFORMER_MODEL)

            self.in_features = EmbeddingService._instance.vector_size

      def train(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 1) -> Dict[str, Any]:

            df = data.copy()
            df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})

            mapeamento_mil = {"bots": 1, "unsafe": 0}
            df["decision_mil"] = df["decision"].map(mapeamento_mil)

            logger.info("Gerando embeddings textuais...")
            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)
            logger.info(f"Agrupando {len(df)} requisições por IPs únicos...")

            logger.info("Agrupando requisições por IP...")
            bags_df = df.groupby("ip").agg({
                  "embedding": list,
                  "decision_mil": list
            }).reset_index()

            bags_df["bag_label"] = bags_df["decision_mil"].apply(lambda labels: 1.0 if 1 in labels else 0.0)

            dataset = MILBagDatasetLogical(bags_df)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            modelo = AttentionMIL(in_features=self.in_features, hidden_dim=self.hidden_dim).to(self.device)
            optimizer = optim.Adam(modelo.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            modelo.train()
            logger.info(f"Iniciando treinamento ({epochs} épocas) no {self.device}...")

            for epoch in range(epochs):
                  loss_acumulada = 0.0
                  for bag, label, _ in dataloader:
                        bag, label = bag.to(self.device), label.to(self.device)
                        
                        optimizer.zero_grad()
                        pred, _ = modelo(bag)
                        loss = criterion(pred, label)
                        loss.backward()
                        optimizer.step()
                        
                        loss_acumulada += loss.item()
                  
                  logger.info(f"Época {epoch+1}/{epochs} - Loss: {loss_acumulada/len(dataloader):.4f}")

            torch.save({
                  "model_state_dict": modelo.state_dict(),
                  "config": {"in_features": self.in_features, "hidden_dim": self.hidden_dim},
            }, self.model_path)

            logger.info(f"Modelo salvo com sucesso em: {self.model_path}")
            return {"status": "success", "loss_final": loss_acumulada/len(dataloader)}

      def predict(self, data: pd.DataFrame) -> pd.DataFrame:

            df = data.copy()

            embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
            df["embedding"] = list(embeddings_matrix)

            bags_df = df.groupby("ip").agg({
                  "embedding": list
            }).reset_index()

            # 3. Carrega o Modelo MIL
            modelo = AttentionMIL(in_features=self.in_features, hidden_dim=self.hidden_dim).to(self.device)
            checkpoint = torch.load(self.model_path, weights_only=False)
            modelo.load_state_dict(checkpoint["model_state_dict"])
            modelo.eval()

            resultados_finais = []

            # 4. Inferência e Desempacotamento (Bag -> Instância)
            with torch.no_grad():
                  for _, row in bags_df.iterrows():
                        ip_atual = row["ip"]
                        bag_tensor = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        # O modelo julga a Bag
                        pred, attention = modelo(bag_tensor)
                        certeza_bag = pred.item()
                        classe_predita = "bots" if certeza_bag > 0.5 else "unsafe"
                        
                        # Prepara os pesos de atenção
                        pesos = attention.squeeze().cpu().numpy()
                        if pesos.ndim == 0: 
                              pesos = [pesos.item()]
                        else: 
                              pesos = pesos.tolist()

                        # A MÁGICA: Puxamos as linhas originais desse IP do DataFrame
                        linhas_do_ip = df[df["ip"] == ip_atual].copy()
                        
                        # Injetamos o veredito final para todas as linhas desse IP
                        linhas_do_ip["pred"] = classe_predita
                        linhas_do_ip["certeza_bag"] = round(certeza_bag * 100, 2)
                        
                        # Injetamos a "culpa" exata (peso de atenção) para cada linha
                        linhas_do_ip["attention_weight"] = [pesos[i] if i < len(pesos) else 0.0 for i in range(len(linhas_do_ip))]
                        
                        resultados_finais.append(linhas_do_ip)
                  
            # 5. Remonta o DataFrame completo com todas as requisições avaliadas
            df_completo = pd.concat(resultados_finais, ignore_index=True)
            
            # 6. Limpeza de Memória: Removemos a matriz pesada antes de devolver
            if "embedding" in df_completo.columns:
                  df_completo = df_completo.drop(columns=["embedding"])

            # Retorna os dados no formato List[Dict] (exatamente como entraram, mas agora com a previsão)
            return df_completo