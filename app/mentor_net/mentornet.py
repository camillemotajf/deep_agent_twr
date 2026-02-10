import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from app.mentor_net.spl import SPLTarget

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Semente {seed} fixada. Resultados agora são reprodutíveis.")

class MentorNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=10,
        num_classes=2,
        label_emb_dim=8,
        epoch_emb_dim=5,
        max_epochs=100
    ):
        super().__init__()

        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ema_alpha = 0.9
        self.running_var_threshold = None
        self.running_loss_threshold = None
        self.spl_computer = SPLTarget(ema_alpha=0.9, device=self.device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=label_emb_dim
        )

        self.epoch_embedding = nn.Embedding(
            num_embeddings=max_epochs,
            embedding_dim=epoch_emb_dim
        )

        combined_dim = (
            hidden_size * 2
            + label_emb_dim
            + epoch_emb_dim
        )

        self.fc1 = nn.Linear(combined_dim, 20)
        self.fc2 = nn.Linear(20, 1)


    def forward(self, history_seq, label, epoch_vec, **kwargs):

        lstm_out, (h_n, c_n) = self.lstm(history_seq)
        lstm_out = lstm_out.squeeze(1)

        out_f = h_n[-2]
        out_b = h_n[-1]
        lstm_features = torch.cat([out_f, out_b], dim=1)

        label = label.long().clamp(0, self.num_classes - 1)
        label_emb = self.label_embedding(label)

        epoch_emb = self.epoch_embedding(epoch_vec.long())

        combined = torch.cat(
            [lstm_features, label_emb, epoch_emb],
            dim=1
        )

        x = self.fc1(combined)
        logits = self.fc2(x)
        weights = torch.sigmoid(logits)

        return weights

    
    def fit_step(self, history_seq, labels, epoch_vec, epoch_idx, raw_losses, 
                 mode='spl', clean_labels=None, optimizer=None):
        if optimizer is None:
            return None

        self.train()
        optimizer.zero_grad()


        if mode == 'dd' and clean_labels is not None:
            v_star = (labels == clean_labels).float().view(-1, 1)
        else:
            with torch.no_grad():
                v_star = self.calculate_target(history_seq, raw_losses, epoch=epoch_idx, labels=labels)

        v_pred = self(history_seq, labels, epoch_vec)
        loss_mentor = F.mse_loss(v_pred, v_star)
        pred_bin = (v_pred > 0.5).float()
        target_bin = (v_star > 0.5).float()
        acc = (pred_bin == target_bin).float().mean()
        
        loss_mentor.backward()
        optimizer.step()

        return loss_mentor.item(), acc.item()
    
    def calculate_target_classic_spl(self, hist_seq, current_losses, epoch, labels):
        spl_score = self.spl_computer.compute(hist_seq, current_losses, epoch)
        spl_score = spl_score.view(-1)

        ncs_values = hist_seq[:, -1, 2].view(-1)
        v_star = torch.zeros_like(ncs_values)
        
        device = current_losses.device

        mask_gold = (ncs_values > 0.7) & (spl_score > 0.5)
        v_star[mask_gold] = 1.0

        mask_trash = (ncs_values < 0.3) | (spl_score < 0.2)
        v_star[mask_trash] = 0.0

        mask_ambiguo = (~mask_gold) & (~mask_trash)

        if mask_ambiguo.any():
            score_combinado = 0.6 * ncs_values[mask_ambiguo] + 0.4 * spl_score[mask_ambiguo]
            v_soft = torch.sigmoid(15 * (score_combinado - 0.5))
            v_star[mask_ambiguo] = v_soft

        return v_star.view(-1, 1).to(device)

   

    def calculate_target(self, hist_seq, current_losses, epoch, labels):
        device = current_losses.device
        
        # 1. Dados
        ncs_values = hist_seq[:, -1, 2].view(-1) 
        losses_flat = current_losses.detach().view(-1)

        loss_threshold = torch.quantile(losses_flat, 0.8) 
        
        v_star = torch.zeros_like(ncs_values)
        mask_bom = (ncs_values > 0.6)
        v_star[mask_bom] = 1.0

        mask_ruim = (ncs_values < 0.4) & (losses_flat > loss_threshold)
        v_star[mask_ruim] = 0.0

        mask_ambiguo = (~mask_bom) & (~mask_ruim)
        
        if mask_ambiguo.any():
            ambiguos = ncs_values[mask_ambiguo]
            v_soft = torch.sigmoid(20 * (ambiguos - 0.5))
            v_star[mask_ambiguo] = v_soft

        return v_star.view(-1, 1).to(device)