import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import hdbscan
import plotly.express as px

class RequestClusteringPipeline:
    def __init__(self, embedding_model, max_header_freq=0.7, tsne_perplexity=30, min_cluster_size=30, min_samples=10):
        """
        Inicializa o pipeline de clusterização e visualização.
        
        :param embedding_model: Modelo base carregado (ex: SentenceTransformer('all-MiniLM-L6-v2'))
        :param max_header_freq: Frequência máxima (0 a 1) para manter um header.
        """
        self.model = embedding_model
        self.max_header_freq = max_header_freq
        self.tsne_perplexity = tsne_perplexity
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
        # Estados internos armazenados após o fit
        self.important_headers = set()
        self.tsne_raw_coords = None
        self.tsne_filtered_coords = None
        self.clusterer = None

    def _filter_headers(self, df):
        print("1. Calculando cardinalidade e filtrando headers...")
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
        print(f"   -> Headers mantidos ({len(self.important_headers)}):", self.important_headers)

        def apply_filter(headers):
            if not isinstance(headers, dict): return {}
            return {k: v for k, v in headers.items() if k in self.important_headers}

        df["filtered_headers"] = df["headers"].apply(apply_filter)
        return df

    def _serialize_text(self, row):
        parts = []
        for k, v in row.get("filtered_headers", {}).items():
            parts.append(f"{k}: {v}")
        
        parts.append("\n[PAYLOAD]") 
        
        request_data = row.get("request", {})
        if isinstance(request_data, dict):
            for k, v in request_data.items():
                val_str = str(v)[:500] 
                parts.append(f"{k}={val_str}")
            
        return "\n".join(parts)

    def preprocess_and_encode(self, df):
        print("2. Serializando requisições e gerando embeddings...")
        df = self._filter_headers(df)
        df["text_filtered"] = df.apply(self._serialize_text, axis=1)
        
        # Padronizando a coluna decision para evitar erros ("bots" -> "bot")
        if "decision" in df.columns:
            df["decision"] = df["decision"].replace({"bots": "bot"})

        print("   -> Gerando embeddings RAW...")
        embeddings_raw = self.model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
        
        print("   -> Gerando embeddings FILTERED...")
        embeddings_filtered = self.model.encode(df["text_filtered"].tolist(), batch_size=64, show_progress_bar=True)
        
        return df, embeddings_raw, embeddings_filtered

    def reduce_dimensions(self, embeddings_raw, embeddings_filtered):
        print("3. Aplicando t-SNE (Redução de dimensionalidade)...")
        tsne = TSNE(n_components=3, perplexity=self.tsne_perplexity, random_state=42)
        
        self.tsne_raw_coords = tsne.fit_transform(embeddings_raw)
        self.tsne_filtered_coords = tsne.fit_transform(embeddings_filtered)

    def cluster_and_classify(self, df, embeddings_filtered):
        print("4. Clusterizando com HDBSCAN e classificando...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        
        df["cluster"] = self.clusterer.fit_predict(embeddings_filtered)
        
        # Mapeamento do cluster para o label majoritário
        cluster_labels_df = (
            df.groupby("cluster")["decision"]
            .value_counts()
            .unstack(fill_value=0)
        )

        cluster_map = cluster_labels_df.apply(
            lambda row: "bot" if row.get("bot", 0) > row.get("unsafe", 0) else "unsafe",
            axis=1
        ).to_dict()

        df["cluster_label"] = df["cluster"].map(cluster_map)

        def classify_row(row):
            if row["decision"] == "unsafe" and row["cluster_label"] == "unsafe":
                return "correct_unsafe"
            elif row["decision"] == "bot" and row["cluster_label"] == "bot":
                return "correct_bot"
            elif row["decision"] == "unsafe" and row["cluster_label"] == "bot":
                return "unsafe_pred_bot"
            elif row["decision"] == "bot" and row["cluster_label"] == "unsafe":
                return "bot_pred_unsafe"
            return "unknown"

        df["classification"] = df.apply(classify_row, axis=1)
        return df

    def visualize(self, df):
        print("5. Gerando visualizações 3D...")
        plot_df = df.copy()
        plot_df["x"] = self.tsne_filtered_coords[:, 0]
        plot_df["y"] = self.tsne_filtered_coords[:, 1]
        plot_df["z"] = self.tsne_filtered_coords[:, 2]

        # Gráfico 1: Todos os dados classificados
        fig_all = px.scatter_3d(
            plot_df, x="x", y="y", z="z", color="classification",
            color_discrete_map={
                "correct_unsafe": "blue",
                "correct_bot": "green",
                "unsafe_pred_bot": "red",
                "bot_pred_unsafe": "orange",
                "unknown": "gray"
            },
            title="t-SNE visualization - clusters vs real labels"
        )
        fig_all.update_traces(marker=dict(size=3, opacity=0.7))
        fig_all.show()

        # Gráfico 2: Apenas os erros
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

    def run(self, df):
        """
        Executa o pipeline completo de ponta a ponta.
        """
        df_processed, emb_raw, emb_filtered = self.preprocess_and_encode(df)
        self.reduce_dimensions(emb_raw, emb_filtered)
        df_final = self.cluster_and_classify(df_processed, emb_filtered)
        self.visualize(df_final)
        
        print("Pipeline finalizado com sucesso!")
        return df_final