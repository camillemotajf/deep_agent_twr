import torch
import torch.nn.functional as F

from app.mentor_net.mentornet import MentorNet
from app.mentor_net.student_mlp import MLPStudent

class MentorNetPredictor:
    def __init__(self, artifact_path, device="cpu"):
        self.device = device
        
        checkpoint = torch.load(artifact_path, map_location=device)
        config = checkpoint["config"]
   
        self.student = MLPStudent(
            input_size=config["embed_dim"], 
            hidden_size=256, 
            output_size=config["num_classes"]
        ).to(device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.student.eval()

        self.mentor = MentorNet(
            input_size=3, 
            hidden_size=32, 
            num_classes=config["num_classes"]
        ).to(device)

        self.mentor.load_state_dict(checkpoint["mentor_state_dict"])
        self.mentor.eval()
        
        self.ref_embeddings = checkpoint["reference_set"]["embeddings"].to(device)
        self.ref_labels = checkpoint["reference_set"]["labels"].to(device)
        
        self.window_size = config["window_size"]

    def predict(self, data_loader):
        """
        Returns a dataframe containing the columns: 
         - target: label from original dataset
         - pred: label predicted by model
         - weight: "mentor trust" of the sample
         - loss: "cust" to predict the label
         - ncs: score of consistence of the sample
        
        """
        results = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["x"].to(self.device)
                targets = batch.get("label", torch.zeros(len(inputs))).to(self.device) 
                logits, emb = self.student(inputs, return_embeddings=True)
    
                raw_losses = self.student.get_individual_losses(logits, targets)
                ncs = self._calculate_ncs_inference(emb, targets)

                loss_val = torch.tanh(raw_losses.view(-1, 1))
                loss_diff = torch.zeros_like(loss_val) 
                
                current_features = torch.cat([loss_val, loss_diff, ncs], dim=1)

                hist_seq = current_features.unsqueeze(1).repeat(1, self.window_size, 1)
                epoch_vec = torch.full((len(inputs),), 99, device=self.device)
                
                weights = self.mentor(hist_seq, label=targets, epoch_vec=epoch_vec)
                preds = torch.argmax(logits, dim=1)
                
                batch_res = torch.stack([
                    targets.float(),    
                    preds.float(),     
                    weights.view(-1),   
                    loss_val.view(-1), 
                    ncs.view(-1)        
                ], dim=1)
                
                results.append(batch_res)

        final_tensor = torch.cat(results)

        if hasattr(data_loader.dataset, 'indices'):
            indices = data_loader.dataset.indices
            df_final = data_loader.dataset.dataset.df.iloc[indices].copy()
        else:
            df_final = data_loader.dataset.df.copy()

        df_final = df_final.reset_index(drop=True)

        df_final["target"] = final_tensor[:, 0].cpu().numpy()
        df_final["pred"]   = final_tensor[:, 1].cpu().numpy()
        df_final["weight"] = final_tensor[:, 2].cpu().numpy() # Confiança do Mentor
        df_final["loss"]   = final_tensor[:, 3].cpu().numpy()
        df_final["ncs"]    = final_tensor[:, 4].cpu().numpy() # Consistência

        return df_final


    def _calculate_ncs_inference(self, query_emb, query_label, k=10):
    
        q_norm = F.normalize(query_emb, p=2, dim=1)
        ref_norm = F.normalize(self.ref_embeddings, p=2, dim=1)
        
        sim_matrix = torch.mm(q_norm, ref_norm.t())
        topk_sim, topk_idx = torch.topk(sim_matrix, k, dim=1)
        
        neighbor_labels = self.ref_labels[topk_idx] # [Batch, K]

        q_lbl_expanded = query_label.view(-1, 1).expand(-1, k)
        matches = (neighbor_labels == q_lbl_expanded).float()
        
        ncs = matches.mean(dim=1).view(-1, 1)
        return ncs