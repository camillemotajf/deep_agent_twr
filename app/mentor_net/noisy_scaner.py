import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

class NoiseScanner:
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        
    def scan_dataset(self, loader, original_dataframe):
    
        self.trainer.student.eval()
        if self.trainer.mentor: self.trainer.mentor.eval()
        
        results = []        
        ref_emb, ref_targets = self.trainer._update_reference_set(loader)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Scanning"):
                ids = batch["id"].cpu().numpy()
                inputs = batch["x"].to(self.device)
                targets = batch["label"].to(self.device)
                
                # 1. Student
                logits, emb = self.trainer.student(inputs, return_embeddings=True)
                raw_losses = self.trainer.student.get_individual_losses(logits, targets)
                preds = logits.argmax(dim=1)
                
                # 2. NCS
                ncs = self.trainer.get_neighborhood_score(emb, targets, k=10)
                
                # 3. Mentor (Simulação Snapshot)
                loss_val = raw_losses.view(-1, 1)
                loss_diff = torch.zeros_like(loss_val) 
                current_features = torch.cat([loss_val, loss_diff, ncs], dim=1)
                hist_seq = current_features.unsqueeze(1).repeat(1, 5, 1) 
                epoch_vec = torch.full((len(targets),), 99, device=self.device)
                
                mentor_trust = torch.ones_like(raw_losses)
                if self.trainer.mentor:
                    mentor_trust = self.trainer.mentor(hist_seq, label=targets, epoch_vec=epoch_vec)
                
                # 4. Coleta (Pegamos TUDO agora)
                for i in range(len(ids)):
                    results.append({
                        'id': ids[i],
                        'loss': raw_losses[i].item(),
                        'ncs': ncs[i].item(),
                        'mentor_trust': mentor_trust[i].item(),
                        'model_pred': preds[i].item(),
                        'true_label': targets[i].item()
                    })
                    
        # Cria DataFrame Bruto
        diag_df = pd.DataFrame(results)
        
        # Merge com dados originais (Texto/JSON)
        if "id" not in original_dataframe.columns:
            original_dataframe["id"] = np.arange(len(original_dataframe))
            
        full_df = pd.merge(original_dataframe, diag_df, on='id', how='inner')
        
        return full_df

    def prepare_for_agent(self, df):
        """
        ETAPA 2: Prepara os dados para o Agente de IA.
        - Faz parsing de JSON (User-Agent, Headers).
        - Calcula Matriz de Confusão Global.
        - Filtra apenas os casos 'Problemáticos' para o Agente analisar.
        """

        # --- A. Feature Engineering (Parsing Seguro de JSON) ---
        def safe_extract(x, key):
            try:
                if isinstance(x, str):
                    # Corrige aspas simples para duplas (comum em logs python)
                    data = json.loads(x.replace("'", '"'))
                else:
                    data = x
                return data.get(key, "MISSING")
            except:
                return "PARSE_ERROR"

        # Extração de colunas críticas para Bot Detection
        # O Agente vai amar ter isso pronto em vez de brigar com JSON
        df['clean_ua'] = df['headers'].apply(lambda x: safe_extract(x, 'User-Agent'))
        df['clean_accept_lang'] = df['headers'].apply(lambda x: safe_extract(x, 'Accept-Language'))
        
        # --- B. Estatísticas Globais (Contexto Situacional) ---
        y_true = df['true_label']
        y_pred = df['model_pred']
        
        # Calcula Matriz de Confusão Global
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        
        disagreement_count = len(df[df['true_label'] != df['model_pred']])
        
        # Foco: Hidden Bots (Label=1 Humano, mas Modelo=0 Bot)
        hidden_bots_count = fn 

        stats_context = f"""
        [GLOBAL DATASET STATISTICS]
        - Total Samples Scanned: {len(df)}
        - Overall Model Accuracy: {acc:.2%}
        - Total Disagreements: {disagreement_count}
        
        [CONFUSION MATRIX BREAKDOWN]
        - True Bots (Correctly Identified): {tn}
        - True Humans (Correctly Identified): {tp}
        - HIDDEN BOTS (Label=Human, Pred=Bot): {fn}  <-- MAIN FOCUS AREA
        - FALSE ALARMS (Label=Bot, Pred=Human): {fp}
        
        [MENTOR INSIGHT]
        - Samples rejected by Mentor (Trust < 0.2): {len(df[df['mentor_trust'] < 0.2])}
        
        [YOUR MISSION]
        The ML Model suggests that {fn} samples labeled as 'Human' are actually BOTS.
        You must analyze the 'clean_ua', 'clean_accept_lang' and 'params' of these specific samples to validate this hypothesis.
        """

        # --- C. Filtragem (O Agente só recebe o problema) ---
        # Filtramos para enviar ao agente apenas:
        # 1. Onde o modelo discorda do rótulo (Erros/Descobertas)
        # 2. OU onde o Mentor rejeitou fortemente (Ruído puro)
        mask_problematic = (df['true_label'] != df['model_pred']) | (df['mentor_trust'] < 0.2)
        
        suspicious_df = df[mask_problematic].copy()
        
        print(f"Dados prontos! Agente receberá {len(suspicious_df)} amostras suspeitas de um total de {len(df)}.")
        
        return suspicious_df, stats_context