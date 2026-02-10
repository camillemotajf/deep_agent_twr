import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPStudent(nn.Module):
    def __init__(self, input_size=384, hidden_size=512, output_size=2, dropout=0.1, bank_size=4096):
        super().__init__()
        
        self._num_classes = output_size
        self.emb_dim = 128 
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size), 
            nn.GELU(),                 
            nn.Dropout(dropout)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)

        )

        self.bank_size = bank_size
        self.register_buffer("feature_bank", torch.randn(bank_size, self.emb_dim))
        self.register_buffer("target_bank", torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.feature_bank = F.normalize(self.feature_bank, dim=1)

    def extract_embeddings(self, x):
        x = self.encoder(x)
        emb = self.projector(x)
        return emb

    def forward(self, x, return_embeddings=False):
        emb = self.extract_embeddings(x)
        logits = self.classifier(emb)
        
        if return_embeddings:
            return logits, emb
        return logits

    def num_classes(self):
        return self._num_classes
    
    def get_individual_losses(self, logits, targets):
        if targets.dtype == torch.float32 or targets.dtype == torch.float64:
            targets = targets.long()
        targets = targets.view(-1)
        return F.cross_entropy(logits, targets, reduction='none')

    def compute_weighted_loss(self, individual_losses, weights_v):
        if weights_v is None:
            return individual_losses.mean()
        weights_v = weights_v.view(-1)
        sum_weights = weights_v.sum() + 1e-8
        return (individual_losses * weights_v).sum() / sum_weights
            
    @torch.no_grad()
    def update_bank(self, new_embeddings, new_targets):
        batch_size = new_embeddings.shape[0]
        ptr = int(self.bank_ptr)
        new_embeddings = F.normalize(new_embeddings, dim=1)
        
        if batch_size + ptr > self.bank_size:
            batch_size = self.bank_size - ptr
            new_embeddings = new_embeddings[:batch_size]
            new_targets = new_targets[:batch_size]
        
        self.feature_bank[ptr:ptr + batch_size] = new_embeddings.detach()
        self.target_bank[ptr:ptr + batch_size] = new_targets.detach()
        ptr = (ptr + batch_size) % self.bank_size
        self.bank_ptr[0] = ptr