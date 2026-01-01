import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=1):  # 4=xyz+1 for cloud_id
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim)
        )
    
    def forward(self, P, Q, mask_p, mask_q):
        # Encode P and Q separately (add cloud ID)
        P_enc = torch.cat([P, torch.zeros_like(P[:, :1])], dim=-1)  # P_cloud=0
        Q_enc = torch.cat([Q, torch.ones_like(Q[:, :1])], dim=-1)   # Q_cloud=1
        
        # Per-point features
        feat_p = self.mlp(P_enc)
        feat_q = self.mlp(Q_enc)
        
        # Global features via max pooling
        global_p = torch.max(feat_p * mask_p.unsqueeze(-1), dim=1)[0]
        global_q = torch.max(feat_q * mask_q.unsqueeze(-1), dim=1)[0]
        
        # Change prediction: difference of globals + per-point
        change_global = torch.abs(global_p - global_q).mean(dim=-1)
        change_p = torch.sigmoid(feat_p[:, :, 0] * mask_p.unsqueeze(-1)).squeeze()
        change_q = torch.sigmoid(feat_q[:, :, 0] * mask_q.unsqueeze(-1)).squeeze()
        
        return change_p, change_q, change_global
