import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns

from app.mentor_net.mentornet import MentorNet
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_mentor_health(debug_data):
    """
    Gera o dashboard visual de saúde do MentorNet.
    """
    # Prepara DataFrame
    df = pd.DataFrame({
        'Peso (v)': debug_data['weights'],
        'Loss': debug_data['losses'],
        'Label Original': debug_data['targets'],
        'Pred': debug_data['preds']
    })
    
    # Coluna de Acerto/Erro para colorir o gráfico
    df['Acertou?'] = df.apply(lambda row: 'Sim' if row['Label Original'] == row['Pred'] else 'Não', axis=1)
    
    # Configurações visuais
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # --- 1. Histograma de Confiança (Roxo) ---
    sns.histplot(
        data=df, x='Peso (v)', kde=True, ax=axes[0],
        color='purple', bins=40, alpha=0.5, edgecolor='black'
    )
    axes[0].axvline(0.1, color='red', linestyle='--', linewidth=2, label='Zona de Rejeição (<0.1)')
    axes[0].set_title("Distribuição de Confiança do Mentor", fontsize=14)
    axes[0].set_ylabel("Contagem")
    axes[0].legend()
    
    # --- 2. Scatter Plot (Loss vs Peso) ---
    sns.scatterplot(
        data=df, x='Loss', y='Peso (v)', hue='Acertou?',
        palette={'Sim': 'royalblue', 'Não': '#ff5555'},
        alpha=0.6, ax=axes[1]
    )
    axes[1].set_title("Correlação: Loss vs Peso (O Mentor penaliza o erro?)", fontsize=14)
    axes[1].set_ylim(-0.05, 1.05)
    
    # --- 3. Boxplot (Viés por Classe) ---
    sns.boxplot(
        data=df, x='Label Original', y='Peso (v)',
        palette="Set2", ax=axes[2], width=0.6,
        linewidth=1.5, fliersize=3
    )
    axes[2].set_title("Confiança Média por Classe (Viés)", fontsize=14)
    axes[2].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

class Trainer:
    """
    Trainer genérico para:
    - Student supervisionado
    - MentorNet online (Self-Paced / Noise-aware)
    - Multiclasse
    """

    def __init__(
        self,
        student,
        history_buffer,
        mentor: MentorNet = None,
        device: str = "cpu"
    ):
        
        if not device:
            self.device = "cpu"
        else:
            self.device = device
      
        self.student = student.to(self.device)
        self.mentor = mentor.to(self.device) if mentor else None
        self.history = history_buffer
        self.curriculum_logs = {
                                    'epoch': [],
                                    'avg_ncs': [],       # Qualidade Média dos Clusters
                                    'avg_weight': [],    # Confiança Média do Mentor
                                    'approval_rate': [], # % de dados aceitos (Peso > 0.8)
                                    'rejection_rate': [] # % de dados rejeitados (Peso < 0.2)
                                }
        
        # assert next(self.mentor.parameters()).device == self.device



    # =========================================================
    # STEP STUDENT
    # =========================================================
    def step_student(self, inputs, targets, weights_v=None):
        logits, embeddings = self.student(inputs, return_embeddings=True)
        raw_losses = self.student.get_individual_losses(logits, targets)

        if weights_v is None:
            loss = raw_losses.mean()
        else:
            loss = self.compute_weighted_loss(raw_losses, weights_v)

        return loss, raw_losses, embeddings

    # =========================================================
    # STEP MENTOR
    # =========================================================
    
    def step_mentor(
        self,
        history_seq,
        raw_losses,
        epoch_idx,
        epoch_vec,
        optimizer_mentor,
        labels
    ):
        if optimizer_mentor is None:
            return None
        
        mentor_loss, acc = self.mentor.fit_step(
            history_seq=history_seq,
            raw_losses=raw_losses,
            epoch_idx=epoch_idx,
            epoch_vec=epoch_vec,
            optimizer=optimizer_mentor,
            labels=labels
        )
        return mentor_loss, acc
    

    def get_neighborhood_score(self, current_embeddings, current_targets,   k=10):
        """
        Calcula NCS usando o Feature Bank do Student.
        Extremamente rápido.
        """
        # Normaliza o batch atual
        current_norm = F.normalize(current_embeddings, p=2, dim=1)
        
        # Pega a memória do banco (já normalizada na inserção)
        ref_norm = self.student.feature_bank.detach().to(self.device)
        ref_targets = self.student.target_bank.detach().to(self.device)
        
        # Similaridade de Cosseno (Matriz Batch x Bank)
        # Ex: [2048 x 128] @ [128 x 4096] -> [2048 x 4096]
        sim_matrix = torch.mm(current_norm, ref_norm.t())
        
        # Pega os Top-K vizinhos
        _, topk_indices = sim_matrix.topk(k, dim=1)
        
        # Verifica se as labels dos vizinhos batem com a label atual
        neighbor_labels = ref_targets[topk_indices] # [Batch, k]
        target_expanded = current_targets.view(-1, 1).expand(-1, k)
        
        correct_neighbors = (neighbor_labels == target_expanded).float()
        ncs = correct_neighbors.mean(dim=1).view(-1, 1)
        
        return ncs
    
    def get_balanced_subset(self, dataset, n=1000):
        """
        Seleciona índices equilibrados de referência para qualquer número de classes
        a partir de um dataset PyTorch usado em DataLoader.

        Args:
            dataset: Dataset PyTorch (pode ser TensorDataset ou custom dataset com __getitem__)
            n: número máximo de amostras por classe

        Returns:
            List[int]: índices embaralhados
        """
        import numpy as np

        # Tenta obter os labels
        try:
            # Se for TensorDataset ou similar
            targets = np.array([dataset[i]["label"] for i in range(len(dataset))])
        except:
            # Caso o dataset tenha .targets
            if hasattr(dataset, "targets"):
                targets = np.array(dataset.targets)
            else:
                raise ValueError("Não foi possível extrair labels do dataset")

        classes = np.unique(targets)
        all_indices = []

        for c in classes:
            class_indices = np.where(targets == c)[0]
            size = min(len(class_indices), n)
            selected = np.random.choice(class_indices, size=size, replace=False)
            all_indices.append(selected)

        combined = np.concatenate(all_indices)
        np.random.shuffle(combined)
        return combined.tolist()
    
    def compute_weighted_loss(self, individual_losses, weights_v):
   
        if weights_v is None:
            return individual_losses.mean()
        
        weights_v = weights_v.view(-1)
        weights_v = torch.clamp(weights_v, min=0.0, max=1.0)
        
        sum_weights = weights_v.sum()
        if sum_weights < 1e-6:
            return torch.tensor(0.0, device=individual_losses.device, requires_grad=True)
            
        weighted_losses = individual_losses * weights_v
        return weighted_losses.sum() / sum_weights

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, train_loader, val_loader=None, epochs=10, burn_in=0.4, 
            opt_student=None, opt_mentor=None, scheduler=None):
        
        import numpy as np # Import seguro
        print(f"Treinando em {self.device}...")
        
        scaler = GradScaler() 
        burn_in_epochs = int(burn_in * epochs)
        logs = {"train_loss": [], "mentor_loss": [], "val_acc": []}
        
        # Logs de Currículo
        if not hasattr(self, 'curriculum_logs'):
            self.curriculum_logs = {
                'epoch': [], 'avg_ncs': [], 'avg_weight': [],
                'approval_rate': [], 'rejection_rate': []
            }

        device_type = self.device.type if isinstance(self.device, torch.device) else self.device
        if 'cuda' in device_type: device_type = 'cuda'

        for epoch in trange(epochs, desc="Epochs"):
            self.student.train()
            if self.mentor: self.mentor.train()

            epoch_loss_student = []
            epoch_loss_mentor = []
            epoch_acc_mentor = []
            temp_ncs, temp_weights, temp_approved, temp_rejected = [], [], [], []

            for batch_idx, batch in enumerate(train_loader):
                indices = batch["id"]
                x = batch["x"].to(self.device)
                y = batch["label"].to(self.device)
                
                with autocast(device_type=device_type):
                    # 1. Student Forward
                    logits, _ = self.student(x, return_embeddings=True)
                    embeddings = self.student.extract_embeddings(x)
                    raw_losses = self.student.get_individual_losses(logits, y)

                    # 2. Update Histórico
                    ncs_scores = self.get_neighborhood_score(embeddings, y, k=10)
                    loss_val = raw_losses.detach().view(-1, 1)
                    loss_diff = loss_val - loss_val.mean()
                    current_features = torch.cat([loss_val, loss_diff, ncs_scores], dim=1)
                    self.history.update(indices, current_features)

                    # 3. MENTOR FORWARD (SEM INTERVENÇÃO MANUAL)
                    # O peso padrão é 1.0 (aceita tudo) até o burn-in acabar
                    weights_v = torch.ones_like(raw_losses)

                    if self.mentor and epoch >= burn_in_epochs:
                        history_seq = self.history.get(indices)
                        epoch_vec = torch.full((len(y),), epoch, device=self.device)

                        with torch.no_grad():
                            # AQUI MUDOU: Confiamos 100% na saída da rede neural
                            weights_v = self.mentor(history_seq, label=y, epoch_vec=epoch_vec).view(-1)
                            
                            # A única intervenção permitida é garantir o intervalo matemático [0, 1]
                            weights_v = torch.clamp(weights_v, 0.0, 1.0)

                            # Coleta métricas para gráficos (Numpy seguro)
                            temp_ncs.append(ncs_scores.mean().item())
                            temp_weights.append(weights_v.mean().item())
                            temp_approved.append((weights_v > 0.8).float().mean().item())
                            temp_rejected.append((weights_v < 0.2).float().mean().item())

                    # 4. Loss Ponderada
                    loss = (raw_losses * weights_v).mean()

                # 5. Otimização Student
                opt_student.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt_student)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                scaler.step(opt_student)
                scaler.update()
                self.student.update_bank(embeddings, y)
                epoch_loss_student.append(loss.item())

                # 6. Otimização Mentor (Aqui ele aprende a imitar o calculate_target)
                if self.mentor and epoch >= burn_in_epochs and opt_mentor:
                    m_loss, m_acc = self.step_mentor(
                        history_seq, raw_losses, epoch, epoch_vec, opt_mentor, y
                    )
                    epoch_loss_mentor.append(m_loss)
                    epoch_acc_mentor.append(m_acc)

            # --- FIM DA ÉPOCA (Logs e Plots) ---
            
            mean_loss_mentor = np.mean(epoch_loss_mentor)
            mean_acc_mentor = np.mean(epoch_acc_mentor)
            loss_mentor_log = ""
            # Atualiza logs de Currículo
            if self.mentor and epoch >= burn_in_epochs and len(temp_ncs) > 0:
                self.curriculum_logs['epoch'].append(epoch)
                self.curriculum_logs['avg_ncs'].append(np.mean(temp_ncs))
                self.curriculum_logs['avg_weight'].append(np.mean(temp_weights))
                self.curriculum_logs['approval_rate'].append(np.mean(temp_approved))
                self.curriculum_logs['rejection_rate'].append(np.mean(temp_rejected))
                loss_mentor_log = f"| Loss Mentor: {mean_loss_mentor:.4f} | Acc Mentor: {mean_acc_mentor}%"

            mean_loss = np.mean(epoch_loss_student)
            logs["train_loss"].append(mean_loss)
            
            val_str = ""
            if val_loader:
                acc = self.evaluate_student(val_loader)
                logs["val_acc"].append(acc)
                val_str = f"| Val Acc: {acc*100:.2f}%"

            status = "[Mentor]" if epoch >= burn_in_epochs else "[Burn-in]"
            print(f"Epoch {epoch+1}/{epochs} {status} | Loss Student: {mean_loss:.4f} {val_str} {loss_mentor_log}")

            if epoch == epochs - 1 and self.mentor:
                print(f"Gerando diagnósticos finais...")
                last_batch_data = {
                    'losses': raw_losses.detach().cpu().numpy(),
                    'ncs': ncs_scores.detach().cpu().numpy(),
                    'weights': weights_v.detach().cpu().numpy(),
                    'targets': y.detach().cpu().numpy(),
                    'preds': logits.argmax(dim=1).detach().cpu().numpy()
                }
                self.plot_mentor_diagnostics(last_batch_data, epoch=epoch+1)

        return logs


    # Helper para criar o reference set
    def _update_reference_set(self, loader):
        with torch.no_grad():
            ref_indices = self.get_balanced_subset(loader.dataset, n=100) # n menor p/ performance
            batch_ref = [loader.dataset[i] for i in ref_indices]
            
            # Ajuste dependendo se seu dataset retorna dict ou tupla
            try:
                ref_inputs = torch.stack([b["x"] for b in batch_ref]).to(self.device)
                ref_targets = torch.tensor([b["label"] for b in batch_ref], device=self.device)
            except: # Fallback genérico
                pass 
            
            ref_embeddings = self.student.extract_embeddings(ref_inputs)
        return ref_embeddings, ref_targets

    # =========================================================
    # EVALUATE MENTOR CORRIGIDO
    # =========================================================
    def evaluate_mentor(self, loader, noisy_indices, epoch=0):
        self.mentor.eval()
        v_clean, v_noisy = [], []
        
        # Precisamos do reference set aqui também para calcular o NCS!
        ref_embeddings, ref_targets = self._update_reference_set(loader)

        with torch.no_grad():
            for batch in loader:
                indices = batch["id"]
                inputs = batch["x"].to(self.device)
                targets = batch["label"].to(self.device)

                # Forward para pegar loss e embeddings
                logits, embeddings = self.student(inputs, return_embeddings=True)
                raw_losses = self.student.get_individual_losses(logits, targets)

                # Calcular features
                loss_val = raw_losses.view(-1, 1)
                loss_diff = loss_val - loss_val.mean()
                
                # --- CORREÇÃO: Calcular NCS ---
                ncs_scores = self.get_neighborhood_score(
                    embeddings, targets, ref_embeddings, ref_targets, k=10
                )

                # Atualizar histórico com AS MESMAS features do treino
                current_features = torch.cat([loss_val, loss_diff, ncs_scores], dim=1)
                self.history.update(indices, current_features)

                # Predição do Mentor
                hist_seq = self.history.get(indices)
                epoch_vec = torch.full((len(targets),), min(int(epoch*99), 99), device=self.device)

                v = self.mentor(hist_seq, targets, epoch_vec).squeeze()

                # Separação Clean/Noisy
                for i, idx in enumerate(indices.cpu().tolist()):
                    val = v[i].item() if v.dim() > 0 else v.item()
                    if idx in noisy_indices:
                        v_noisy.append(val)
                    else:
                        v_clean.append(val)

        return v_clean, v_noisy

    # =========================================================
    # EVALUATE STUDENT
    # =========================================================
    def evaluate_student(self, loader):
        self.student.train()
        correct, total = 0, 0
        all_preds = [] # Debug
        all_targets = [] # Debug

        with torch.no_grad():
            for i, batch in enumerate(loader):

                targets = batch["label"].to(self.device)
                inputs = batch["x"].to(self.device)

                logits = self.student(inputs)
                preds = logits.argmax(dim=1)

                if i == 0:
                    print(f"\n[DEBUG VALIDAÇÃO] Batch 0")
                    print(f"  -> Predições: {preds[:10].tolist()}")
                    print(f"  -> Reais    : {targets[:10].tolist()}")

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                all_preds.append(preds)
                all_targets.append(targets)
        # --- DEBUG CRÍTICO ---
        flat_preds = torch.cat(all_preds)
        flat_targets = torch.cat(all_targets)
        unique, counts = torch.unique(flat_preds, return_counts=True)
        print(f"\n[DEBUG] Distribuição de Predições: {dict(zip(unique.tolist(), counts.tolist()))}")
        print(f"[DEBUG] Distribuição Real (Targets): {dict(zip(*torch.unique(flat_targets, return_counts=True)))}")

        acc = correct / total if total > 0 else 0
        
        print(f"[DEBUG MODE: FORCE TRAIN] Val Acc: {acc*100:.2f}%")
        # ---------------------
        return acc

    def diagnose_mentor_performance(self, loader):
        """
        Executa um check-up completo no MentorNet:
        1. Calcula métricas estatísticas (Correlação, Viés, Rejeição).
        2. Gera os gráficos de diagnóstico.
        """
        print("\n--- 🩺 Iniciando Diagnóstico de Saúde do Mentor ---")
        
        self.student.eval()
        if self.mentor: self.mentor.eval()
        
        # Listas para coleta de dados
        all_losses = []
        all_weights = []
        all_targets = []
        all_preds = []
        
        # Atualiza referência para cálculo correto do NCS
        print("   -> Atualizando conjunto de referência para NCS...")
        ref_embeddings, ref_targets = self._update_reference_set(loader)

        with torch.no_grad():
            for batch in loader:
                ids = batch["id"]
                inputs = batch["x"].to(self.device)
                targets = batch["label"].to(self.device)

                # 1. Student Forward
                logits, emb = self.student(inputs, return_embeddings=True)
                raw_losses = self.student.get_individual_losses(logits, targets)
                preds = logits.argmax(dim=1)
                
                # 2. Engenharia de Features (Input do Mentor)
                # NCS (Coerência de Vizinhança)
                ncs = self.get_neighborhood_score(emb, targets, k=10)
                
                # Loss suavizada (Tanh) e Delta Loss
                loss_val = torch.tanh(raw_losses.view(-1, 1))
                loss_diff = loss_val - loss_val.mean()
                
                # Atualiza histórico temporariamente
                current_features = torch.cat([loss_val, loss_diff, ncs], dim=1)
                self.history.update(ids, current_features)
                hist_seq = self.history.get(ids)
                
                # 3. Mentor Forward
                # Simulamos época 99 (futuro) para ver o comportamento "maduro" do Mentor
                epoch_vec = torch.full((len(targets),), 99, device=self.device)
                
                weights = torch.ones_like(raw_losses) # Default
                if self.mentor:
                    weights = self.mentor(hist_seq, label=targets, epoch_vec=epoch_vec)
                    # Clamp simples (0 a 1), sem normalização por média!
                    weights = torch.clamp(weights.view(-1), 0, 1)

                    min_w = weights.min()
                    max_w = weights.max()

                    # Evita divisão por zero
                    if max_w - min_w > 1e-4:
                        weights = (weights - min_w) / (max_w - min_w)

                # 4. Coleta
                all_losses.extend(raw_losses.cpu().numpy())
                all_weights.extend(weights.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # --- CÁLCULO DAS MÉTRICAS ---
        all_losses = np.array(all_losses)
        all_weights = np.array(all_weights)
        print("ALL WEIGHTS: ", all_weights)
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)

        # 1. Correlação (Loss x Peso) -> Esperado: Negativa (Loss Alta = Peso Baixo)
        correlation = np.corrcoef(all_losses, all_weights)[0, 1]
        
        # 2. Taxa de Rejeição -> Esperado: 5% a 20%
        rejection_rate = np.mean(all_weights < 0.1)
        
        # 3. Viés de Classe -> Esperado: Médias próximas
        mean_w_human = np.mean(all_weights[all_targets == 0]) if (all_targets == 0).any() else 0
        mean_w_bot = np.mean(all_weights[all_targets == 1]) if (all_targets == 1).any() else 0

        # --- RELATÓRIO TEXTUAL ---
        print("-" * 50)
        print(f"MÉTRICAS VITAIS:")
        
        print(f"1. Correlação (Loss x Peso): {correlation:.4f}")
        if correlation < -0.5: print("   ✅ ÓTIMO. O Mentor penaliza erros.")
        elif correlation < 0:  print("   ⚠️ FRACO. O Mentor está tímido.")
        else:                  print("   ❌ CRÍTICO. O Mentor está ajudando o erro (Correlação Positiva).")

        print(f"2. Taxa de Rejeição (<0.1):  {rejection_rate*100:.2f}%")
        if 0.05 < rejection_rate < 0.3: print("   ✅ SAUDÁVEL. Filtra ruído sem matar o dataset.")
        elif rejection_rate > 0.5:      print("   ❌ AGRESSIVO. Rejeita mais da metade dos dados.")
        else:                           print("   ⚠️ PERMISSIVO. Aceita quase tudo.")

        print(f"3. Confiança Média por Classe:")
        print(f"   - Classe 0 (Humanos): {mean_w_human:.4f}")
        print(f"   - Classe 1 (Bots):    {mean_w_bot:.4f}")
        if abs(mean_w_human - mean_w_bot) > 0.3:
            print("   ❌ ALERTA DE VIÉS. O Mentor desistiu de uma das classes.")
        else:
            print("   ✅ EQUILIBRADO.")
        print("-" * 50)

        # --- CHAMADA DA PLOTAGEM ---
        plot_data = {
            'weights': all_weights,
            'losses': all_losses,
            'targets': all_targets,
            'preds': all_preds
        }
        
        print("Gerando gráficos...")
        plot_mentor_health(plot_data)

        
        return correlation, rejection_rate

    def plot_mentor_diagnostics(self, debug_data, epoch):
        """
        Gera o dashboard visual de saúde do MentorNet.
        BLINDADO: Converte automaticamente qualquer Tensor CUDA para Numpy CPU.
        """
        # Imports locais para portabilidade
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import torch

        # --- FUNÇÃO DE SEGURANÇA (O Porteiro) ---
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy() # Traz da GPU para CPU
            if isinstance(x, list):
                return np.array(x)
            return x # Se já for numpy, deixa passar

        # 1. Extração e Conversão Segura
        losses = to_numpy(debug_data['losses']).flatten()
        weights = to_numpy(debug_data['weights']).flatten()
        targets = to_numpy(debug_data['targets']).flatten()
        
        # NCS e Preds (com fallback caso não existam no dict)
        if 'ncs' in debug_data:
            ncs = to_numpy(debug_data['ncs']).flatten()
        else:
            ncs = np.zeros_like(losses) # Fallback para não quebrar

        if 'preds' in debug_data:
            preds = to_numpy(debug_data['preds']).flatten()
        else:
            preds = np.zeros_like(targets)

        # 2. Criação do DataFrame (100% CPU)
        df = pd.DataFrame({
            'Peso (v)': weights,
            'Loss': losses,
            'Label Original': targets,
            'NCS': ncs,
            'Pred': preds
        })
        
        df['Acertou?'] = np.where(df['Label Original'] == df['Pred'], 'Sim', 'Não')

        # 3. Plotagem
        plt.close('all') 
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # --- Gráfico 1: Distribuição de Loss (ou NCS se preferir) ---
        try:
            sns.kdeplot(data=df, x='Loss', hue='Label Original', fill=True, 
                        palette=['blue', 'red'], ax=axes[0], warn_singular=False)
        except:
            # Fallback se a variância for 0
            sns.histplot(data=df, x='Loss', hue='Label Original', ax=axes[0])
        axes[0].set_title("1. Distribuição de Loss (Dificuldade)")
        
        # --- Gráfico 2: Scatter (Decisão) ---
        sns.scatterplot(
            data=df, x='Loss', y='Peso (v)', 
            hue='Acertou?', style='Label Original',
            palette={'Sim': 'cornflowerblue', 'Não': 'red'},
            alpha=0.6, ax=axes[1]
        )
        axes[1].axhline(0.5, color='gray', linestyle='--')
        axes[1].set_title("2. Decisão do Mentor")
        axes[1].set_ylabel("Peso Atribuído (v)")
        axes[1].set_ylim(-0.05, 1.05)

        # --- Gráfico 3: Boxplot (Viés) ---
        # Este é crucial para ver se o Mentor está matando uma classe específica
        sns.boxplot(
            data=df, x='Label Original', y='Peso (v)',
            palette="Set2", ax=axes[2], width=0.5,
            linewidth=1.5, fliersize=3
        )
        axes[2].set_title("3. Confiança Média por Classe (Viés)")
        axes[2].set_ylim(-0.05, 1.05)

        plt.suptitle(f"Diagnóstico MentorNet - Época {epoch}", fontsize=16)
        plt.tight_layout()
        plt.show()