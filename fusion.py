import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    """
    α^m = sigmoid(h^m, W_α)
    β^m = sigmoid(h_hat^m, W_β)
    h^m_i = h_hat^m_i                           if modality m is missing
          = α^m_i · h^m_i + β^m_i · h_hat^m_i   otherwise
    """

    def __init__(self, hidden_dim, mode='gated'):
        super().__init__()
        self.mode = mode
        self.W_alpha = nn.Linear(hidden_dim, 1)
        self.W_beta = nn.Linear(hidden_dim, 1)

    def forward(self, h_m_dict, h_hat_m_dict, mask_dict):
        h_fused_dict = {}
        for m in sorted(h_m_dict.keys()):
            h_m = h_m_dict[m]
            h_hat_m = h_hat_m_dict[m]
            mask = mask_dict[m].unsqueeze(-1).float()

            if self.mode == 'average':
                # Equal-weight fusion; missing → h_hat_m only
                fused = 0.5 * h_m + 0.5 * h_hat_m
                h_fused_m = h_hat_m * (1 - mask) + fused * mask
            else:  # 'gated' (default)
                alpha_m = torch.sigmoid(self.W_alpha(h_m))
                beta_m  = torch.sigmoid(self.W_beta(h_hat_m))

                # Normalize to sum to 1
                total = alpha_m + beta_m + 1e-8
                alpha_m_norm = alpha_m / total
                beta_m_norm  = beta_m  / total

                # Fusion: missing → use h_hat^m; present → α·h^m + β·h_hat^m
                h_fused_m = h_hat_m * (1 - mask) + (alpha_m_norm * h_m + beta_m_norm * h_hat_m) * mask

            h_fused_dict[m] = h_fused_m

        return h_fused_dict