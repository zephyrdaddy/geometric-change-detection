import torch
import torch.nn as nn
import torch.nn.functional as F

class ChangeNet(nn.Module):
    """Bidirectional change detection for point clouds P -> Q."""
    def __init__(self, in_dim=2, hidden_dim=128):
        super().__init__()
        # Shared point encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Decoder for per-Q-point change
        self.decoder_q = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Decoder for per-P-point removal
        self.decoder_p = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, P, Q, mask_p=None, mask_q=None):
        """
        P: [B, n_p, 2]
        Q: [B, n_q, 2]
        Returns:
            change_q_logits: [B, n_q]
            change_p_logits: [B, n_p]
        """
        B, n_p, _ = P.shape
        n_q = Q.shape[1]

        # Encode points
        feat_P = self.encoder(P)  # [B, n_p, D]
        feat_Q = self.encoder(Q)  # [B, n_q, D]
        # If mask is provided, fill padding with a very small number
        if mask_p is not None:
            # unsqueeze mask to [B, n_p, 1] to match feat_P [B, n_p, D]
            feat_P_masked = feat_P.masked_fill(mask_p.unsqueeze(-1) == 0, -1e9)
        else:
            feat_P_masked = feat_P

        if mask_q is not None:
            feat_Q_masked = feat_Q.masked_fill(mask_q.unsqueeze(-1) == 0, -1e9)
        else:
            feat_Q_masked = feat_Q



        # Global summaries (Max pooling)
        g_P = torch.max(feat_P_masked, dim=1)[0]  # [B, D]
        g_Q = torch.max(feat_Q_masked, dim=1)[0]  # [B, D]

        # 3. Broadcast global diff
        global_diff_q = (g_Q - g_P).unsqueeze(1).repeat(1, n_q, 1)
        global_diff_p = (g_P - g_Q).unsqueeze(1).repeat(1, n_p, 1)

        # 4. Per-point predictions
        change_q_logits = self.decoder_q(torch.cat([feat_Q, global_diff_q], dim=-1)).squeeze(-1)
        change_p_logits = self.decoder_p(torch.cat([feat_P, global_diff_p], dim=-1)).squeeze(-1)

        return change_q_logits, change_p_logits
        # # Global features
        # g_P = torch.max(feat_P, dim=1)[0]  # [B, D]
        # g_Q = torch.max(feat_Q, dim=1)[0]  # [B, D]

        # # Broadcast global diff
        # global_diff_q = (g_Q - g_P).unsqueeze(1).repeat(1, n_q, 1)
        # global_diff_p = (g_P - g_Q).unsqueeze(1).repeat(1, n_p, 1)

        # # Per-point predictions
        # change_q_logits = self.decoder_q(torch.cat([feat_Q, global_diff_q], dim=-1)).squeeze(-1)
        # change_p_logits = self.decoder_p(torch.cat([feat_P, global_diff_p], dim=-1)).squeeze(-1)

        # return change_q_logits, change_p_logits
