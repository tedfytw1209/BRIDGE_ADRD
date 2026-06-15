import torch
import torch.nn as nn


class ModalityDecoder(nn.Module):
    """
    Peer-context feature-embedding decoder.

    For each feature j, concatenates:
      - the patient GCN embedding (cross-patient context)
      - observed peer feature values for this modality (0 where missing/masked)
      - observed peer feature mask (1=observed, 0=missing/masked)
      - a learnable feature embedding for j (feature identity)
    then passes the combined vector through a shared MLP to produce a scalar.

    This gives the decoder direct access to peer feature values same as LR/KNN while retaining the GCN's cross-patient context.
    Input size: hidden_dim + output_dim + output_dim + hidden_dim = 2*hidden_dim + 2*output_dim
    """

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        # one learnable embedding per output feature (feature identity)
        self.feature_embs = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        # shared MLP: [h || x_obs || feat_mask || feature_emb_j] → scalar
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, x_obs, feat_mask):
        # h: (B, hidden_dim)  - GCN embedding
        # x_obs: (B, output_dim)  - peer feature values (0 where missing/masked)
        # feat_mask: (B, output_dim)  - 1=observed, 0=missing/masked
        B, D = h.size(0), self.feature_embs.size(0)
        h_exp = h.unsqueeze(1).expand(B, D, -1)
        f_exp = self.feature_embs.unsqueeze(0).expand(B, D, -1)
        x_exp = x_obs.unsqueeze(1).expand(B, D, -1)
        m_exp = feat_mask.unsqueeze(1).expand(B, D, -1)
        pairs = torch.cat([h_exp, x_exp, m_exp, f_exp], dim=-1)  # (B, D, 2*hidden_dim + 2*output_dim)
        return self.mlp(pairs).squeeze(-1)