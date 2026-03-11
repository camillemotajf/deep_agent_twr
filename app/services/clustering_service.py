import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import hdbscan
import plotly.express as px
import torch
from app.utils.analise import parse_dict_col
import logging

# Desativa logs de nível INFO das bibliotecas de rede e do Hugging Face
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class RequestClusteringPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", max_header_freq=0.7, tsne_perplexity=30, min_cluster_size=30, min_samples=10, cluster_threshold=0.6):
        """
        Inicializa o pipeline de clusterização e visualização.
        
        :param embedding_model: Modelo base carregado (ex: SentenceTransformer('all-MiniLM-L6-v2'))
        :param max_header_freq: Frequência máxima (0 a 1) para manter um header.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(embedding_model_name, device=self.device)
        self.model = model
        self.max_header_freq = max_header_freq
        self.tsne_perplexity = tsne_perplexity
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_threshold = cluster_threshold
        
        # Estados internos armazenados após o fit
        self.important_headers = set()
        self.tsne_raw_coords = None
        self.tsne_filtered_coords = None
        self.clusterer = None

    def _filter_headers(self, df):
        n_samples = len(df)
        value_sets = defaultdict(set)

        for headers in df["headers"]:
            if isinstance(headers, dict):
                for k, v in headers.items():
                    value_sets[k].add(str(v).lower())

        self.important_headers = {
            k for k, values in value_sets.items()
            if len(values) < self.max_header_freq * n_samples
        }

        def apply_filter(headers):
            if not isinstance(headers, dict): return {}
            return {k: v for k, v in headers.items() if k in self.important_headers and not "user-agent" in k.lower()}

        df["filtered_headers"] = df["headers"].apply(apply_filter)
        return df

    def _serialize_text(self, row, col):
        parts = []
        for k, v in row.get(col, {}).items():
            parts.append(f"{k}: {v}")
        
        parts.append("\n[PAYLOAD]") 
        
        request_data = row.get("request", {})
        if isinstance(request_data, dict):
            for k, v in request_data.items():
                val_str = str(v)[:500] 
                parts.append(f"{k}={val_str}")
            
        return "\n".join(parts)

    def preprocess_and_encode(self, df):

        df = self._filter_headers(df)
        df["text_filtered"] = df.apply(self._serialize_text, axis=1, args=("filtered_headers",))
        # df["text"] = df.apply(self._serialize_text, axis=1, args=("headers",))
        
        # Padronizando a coluna decision para evitar erros ("bots" -> "bot")
        if "decision" in df.columns:
            df["decision"] = df["decision"].replace({"bots": "bot"})

        # print("   -> Gerando embeddings RAW...")
        # embeddings_raw = self.model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
        
        embeddings_filtered = self.model.encode(df["text_filtered"].tolist(), batch_size=64, show_progress_bar=True)
        
        # return df, embeddings_raw, embeddings_filtered
        return df, embeddings_filtered

    def reduce_dimensions(self, embeddings_filtered):
        tsne = TSNE(n_components=3, perplexity=self.tsne_perplexity, random_state=42)
        
        # self.tsne_raw_coords = tsne.fit_transform(embeddings_raw)
        self.tsne_filtered_coords = tsne.fit_transform(embeddings_filtered)

    def cluster_and_classify(self, df, embeddings_filtered):


        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        
        print("Clustering...")
        df["cluster"] = self.clusterer.fit_predict(embeddings_filtered)
        
        # Se não tiver a coluna decision (ground truth), não há como aplicar a regra
        if "decision" not in df.columns:
            df["cluster_label"] = "mixed"
            df["classification"] = "mixed"
            return df

        # Conta a quantidade de cada label ("bot", "unsafe", etc) dentro de cada cluster
        cluster_labels_df = (
            df.groupby("cluster")["decision"]
            .value_counts()
            .unstack(fill_value=0)
        )

        # ---------------------------------------------------------
        # NOVA REGRA DE THRESHOLD (LIMIAR)
        # ---------------------------------------------------------
        def apply_threshold(row):
            total = row.sum()
            if total == 0:
                return "outlier"
            
            bot_ratio = row.get("bot", 0) / total
            unsafe_ratio = row.get("unsafe", 0) / total
            
            # Ex: Se cluster_threshold for 0.7, precisa ter >= 70% de dominância
            if bot_ratio >= self.cluster_threshold:
                return "bot"
            elif unsafe_ratio >= self.cluster_threshold:
                return "unsafe"
            else:
                return "mixed" # Não passou do threshold = cluster misto

        # Aplica a regra para gerar o dicionário de mapeamento
        cluster_map = cluster_labels_df.apply(apply_threshold, axis=1).to_dict()
        
        # O cluster -1 (ruído do HDBSCAN) deve ser sempre tratado à parte
        if -1 in cluster_map:
            cluster_map[-1] = "noise_anomaly"

        # Atribui o rótulo final a cada linha do DataFrame
        df["cluster_label"] = df["cluster"].map(cluster_map)

        # ---------------------------------------------------------
        # MATRIZ DE CONFUSÃO (Erros e Acertos individuais)
        # ---------------------------------------------------------
        def classify_row(row):
            decision = row.get("decision")
            c_label = row.get("cluster_label")
            
            # Ignora o cálculo de erro para clusters mistos, desconhecidos ou ruído
            if pd.isna(decision) or c_label in ["mixed", "mixed", "noise_anomaly"]:
                return "mixed"
                
            if decision == "unsafe" and c_label == "unsafe":
                return "correct_unsafe"
            elif decision == "bot" and c_label == "bot":
                return "correct_bot"
            elif decision == "unsafe" and c_label == "bot":
                return "unsafe_pred_bot"
            elif decision == "bot" and c_label == "unsafe":
                return "bot_pred_unsafe"
                
            return "mixed"

        df["classification"] = df.apply(classify_row, axis=1)
        return df

    # def cluster_and_classify(self, df, embeddings_filtered):
    #     print("4. Clusterizando com HDBSCAN e classificando...")
    #     self.clusterer = hdbscan.HDBSCAN(
    #         min_cluster_size=self.min_cluster_size,
    #         min_samples=self.min_samples
    #     )
        
    #     df["cluster"] = self.clusterer.fit_predict(embeddings_filtered)
        
    #     # Mapeamento do cluster para o label majoritário
    #     cluster_labels_df = (
    #         df.groupby("cluster")["decision"]
    #         .value_counts()
    #         .unstack(fill_value=0)
    #     )

    #     cluster_map = cluster_labels_df.apply(
    #         lambda row: "bot" if row.get("bot", 0) > row.get("unsafe", 0) else "unsafe",
    #         axis=1
    #     ).to_dict()

    #     df["cluster_label"] = df["cluster"].map(cluster_map)

    #     def classify_row(row):
    #         if row["decision"] == "unsafe" and row["cluster_label"] == "unsafe":
    #             return "correct_unsafe"
    #         elif row["decision"] == "bot" and row["cluster_label"] == "bot":
    #             return "correct_bot"
    #         elif row["decision"] == "unsafe" and row["cluster_label"] == "bot":
    #             return "unsafe_pred_bot"
    #         elif row["decision"] == "bot" and row["cluster_label"] == "unsafe":
    #             return "bot_pred_unsafe"
    #         return "mixed"

    #     df["classification"] = df.apply(classify_row, axis=1)
    #     return df

    def visualize(self, df):
        plot_df = df.copy()
        plot_df["x"] = self.tsne_filtered_coords[:, 0]
        plot_df["y"] = self.tsne_filtered_coords[:, 1]
        plot_df["z"] = self.tsne_filtered_coords[:, 2]

        # NOVO GRÁFICO: Dados classificados pela coluna "decision"
        if "decision" in plot_df.columns:
            fig_decision = px.scatter_3d(
                plot_df, x="x", y="y", z="z", color="decision",
                title="t-SNE visualization - by Decision Label"
            )
            fig_decision.update_traces(marker=dict(size=3, opacity=0.7))
            fig_decision.show()
        else:
            print("Aviso: A coluna 'decision' não foi encontrada no DataFrame.")

        # Gráfico 1: Todos os dados classificados (Original)
        fig_all = px.scatter_3d(
            plot_df, x="x", y="y", z="z", color="classification",
            color_discrete_map={
                "correct_unsafe": "blue",
                "correct_bot": "green",
                "unsafe_pred_bot": "red",
                "bot_pred_unsafe": "orange",
                "mixed": "gray"
            },
            title="t-SNE visualization - clusters vs real labels"
        )
        fig_all.update_traces(marker=dict(size=3, opacity=0.7))
        fig_all.show()

        # Gráfico 2: Apenas os erros (Original)
        errors_df = plot_df[plot_df["classification"].isin(["unsafe_pred_bot", "bot_pred_unsafe"])]
        if not errors_df.empty:
            fig_errors = px.scatter_3d(
                errors_df, x="x", y="y", z="z", color="classification",
                color_discrete_map={"unsafe_pred_bot": "red", "bot_pred_unsafe": "orange"},
                title="t-SNE - misclassified samples"
            )
            fig_errors.update_traces(marker=dict(size=3, opacity=0.9))
            fig_errors.show()
        else:
            print("Nenhum erro de classificação encontrado para plotar.")

    def run(self, path):
        df = pd.read_parquet(path)
        df['headers'] = df['headers'].apply(parse_dict_col)
        df['request'] = df['request'].apply(parse_dict_col)

        # df_processed, emb_raw, emb_filtered = self.preprocess_and_encode(df)
        df_processed, emb_filtered = self.preprocess_and_encode(df)
        self.reduce_dimensions(emb_filtered)
        df_final = self.cluster_and_classify(df_processed, emb_filtered)
        self.visualize(df_final)
        return df_final
    
    # def analyze_traffic_for_llm(self) -> str:
    #     try:
    #         # 1. Initial Validation and Processing
    #         if self.df is None or self.df.empty:
    #             return json.dumps({"status": "error", "message": "No dataframe (self.df) loaded for analysis."})
                
    #         df = self.df
    #         self.df = None # Clear memory
            
    #         # Unpacking the 3 variables returned by your specific preprocess_and_encode pipeline
    #         df_processed, emb_raw, emb_filtered = self.preprocess_and_encode(df)
    #         df_final = self.cluster_and_classify(df_processed, emb_filtered)
            
    #         # 2. Metrics Extraction
    #         total_requests = len(df_final)
    #         clusters_found = [c for c in df_final['cluster'].unique() if c != -1]
    #         noise_count = int((df_final['cluster'] == -1).sum())
            
    #         unsafe_pred_bot_count = int((df_final["classification"] == "unsafe_pred_bot").sum())
    #         bot_pred_unsafe_count = int((df_final["classification"] == "bot_pred_unsafe").sum())
            
    #         # 3. Detailed Cluster Analysis
    #         cluster_details = {}
    #         for c in clusters_found:
    #             c_data = df_final[df_final['cluster'] == c]
    #             label = c_data['cluster_label'].iloc[0]
    #             distribution = c_data['decision'].value_counts().to_dict() if 'decision' in c_data.columns else {}
                
    #             # Make the JSON self-explanatory regarding what this cluster represents
    #             cluster_details[f"cluster_{c}"] = {
    #                 "assigned_label": label,
    #                 "cluster_size": len(c_data),
    #                 "actual_data_distribution": distribution,
    #                 "llm_insight": f"This group is predominantly composed of '{label}' traffic."
    #             }

    #         # 4. Anomaly/Noise Samples Collection
    #         anomalies_df = df_final[df_final['cluster'] == -1]
    #         sample_anomalies = []
    #         if not anomalies_df.empty:
    #             # Get up to 3 samples for the LLM to inspect
    #             for _, row in anomalies_df.head(3).iterrows():
    #                 sample_anomalies.append({
    #                     "original_label": row.get("decision", "mixed"),
    #                     "captured_headers": row.get("headers", {}),
    #                     "suspicious_payload": row.get("request", {})
    #                 })

    #         # 5. The "Explanatory" JSON Payload (Context-focused structure for the Agent)
    #         report = {
    #             "status": "success",
    #             "report_objective": "Semantic analysis of HTTP traffic to identify attack patterns and botnets.",
    #             "overview": {
    #                 "total_requests_analyzed": total_requests,
    #                 "behavioral_patterns_found": len(clusters_found),
    #                 "isolated_anomalous_requests": noise_count
    #             },
    #             "critical_security_alerts": {
    #                 "attacks_camouflaged_as_bots": {
    #                     "count": unsafe_pred_bot_count,
    #                     "explanation": "Malicious ('unsafe') requests that have identical structure and headers to automation bots. Strong indication of an attacker using automated tools."
    #                 },
    #                 "bots_grouped_with_attacks": {
    #                     "count": bot_pred_unsafe_count,
    #                     "explanation": "Bot traffic that was structurally confused with manual/targeted attacks."
    #                 }
    #             },
    #             "cluster_details": cluster_details,
    #             "anomaly_samples_for_inspection": sample_anomalies
    #         }
            
    #         return json.dumps(report, indent=2, ensure_ascii=False)
            
    #     except Exception as e:
    #         return json.dumps({
    #             "status": "error", 
    #             "message": "Failed to execute the clustering pipeline.",
    #             "error_details": str(e)
    #         })

    # def analyze_traffic_for_llm(self) -> str:
    #     try:
    #         # 1. Validação inicial
    #         if self.df is None or self.df.empty:
    #             return json.dumps({"status": "error", "message": "Nenhum dataframe carregado para análise."})
                
    #         df = self.df
    #         self.df = None # Limpa a memória
            
    #         # 2. Executa o pipeline
    #         df_processed, emb_raw, emb_filtered = self.preprocess_and_encode(df)
    #         df_final = self.cluster_and_classify(df_processed, emb_filtered)
            
    #         # 3. Extrai as métricas simplificadas
    #         total_samples = len(df_final)
            
    #         # Conta quantas amostras receberam cada rótulo de cluster
    #         contagem_labels = df_final["cluster_label"].value_counts().to_dict()
            
    #         # 4. Monta o payload JSON enxuto
    #         report = {
    #             "status": "success",
    #             "total_samples": total_samples,
    #             "samples_by_cluster_type": {
    #                 "bot": contagem_labels.get("bot", 0),
    #                 "unsafe": contagem_labels.get("unsafe", 0),
    #                 "mixed": contagem_labels.get("mixed", 0),
    #                 # O ruído/anomalia do HDBSCAN (-1)
    #                 "noise_anomaly": contagem_labels.get("noise_anomaly", 0),
    #                 # Caso falte a coluna decision e ele não consiga classificar
    #                 "mixed": contagem_labels.get("mixed", 0) 
    #             }
    #         }
            
    #         # Remove chaves zeradas se quiser deixar ainda mais limpo (opcional)
    #         report["samples_by_cluster_type"] = {k: v for k, v in report["samples_by_cluster_type"].items() if v > 0}
            
    #         return json.dumps(report, indent=2, ensure_ascii=False)
            
    #     except Exception as e:
    #         return json.dumps({
    #             "status": "error", 
    #             "message": str(e)
    #         })

    def filter_investiogation_data(self, df_final):
        mask = (
                df_final["classification"].isin(["unsafe_pred_bot"])
                 | df_final["classification"].isin(["bot_pred_unsafe"])
                #  | df_final["cluster_label"].isin(["mixed", "noise_anomaly"])
            )

        df_investigacao = df_final[mask].copy()
        
        colunas_uteis = ["decision", "cluster", "cluster_label", "classification", "headers", "request", "ip_api_isp"]
        df_investigacao = df_investigacao[colunas_uteis]
        df_investigacao = df_final[mask].copy()

        return df_investigacao

    def analyze_traffic_for_llm(self, filepath) -> str:
        try:
                
            df = pd.read_parquet(filepath)

            df["headers"] = df["headers"].apply(parse_dict_col)
            df["request"] = df["request"].apply(parse_dict_col)

            if df is None or df.empty:
                return json.dumps({"status": "error", "message": "Nenhum dataframe carregado para análise."})
            
            df_processed, emb_filtered = self.preprocess_and_encode(df)
            df_final = self.cluster_and_classify(df_processed, emb_filtered)
            filepath_tosave = filepath.replace("raw_", "processed_")

            df_invest = self.filter_investiogation_data(df_final)
            df_invest.to_parquet(filepath_tosave)
            
            contagem_classificacao = df_final["classification"].value_counts().to_dict()
            
            report = {
                "status": "success",
                "total_samples": len(df_final),
                "classification_counts": contagem_classificacao,
                "investigation_file_path": filepath_tosave, # Passa o novo caminho dinâmico para o LLM
                "investigation_targets_count": len(df_invest)
            }
            
            return json.dumps(report, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "status": "error", 
                "message": str(e)
            })