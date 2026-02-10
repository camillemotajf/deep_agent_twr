import torch

class HistoryBuffer:
    def __init__(self, num_samples, window_size=10, feature_dim=2):
        self.window_size = window_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer = torch.zeros((num_samples, window_size, feature_dim), device=self.device)
        
    def update(self, indices, new_values):
        new_values = new_values.to(self.device)
        if new_values.dim() == 1:
            new_values = new_values.unsqueeze(1) 

        old_hist = self.buffer[indices, 1:]  
        self.buffer[indices] = torch.cat([old_hist, new_values.unsqueeze(1)], dim=1)
        
    def get(self, indices):

        return self.buffer[indices]