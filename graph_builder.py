import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim * 2, hidden_dim)

    def forward(self, x_m, feat_mask):
        """Extract feature representation h_m for modality m.
        h^m_n = f^m_Φ(x^m_n)"""
        x_in = torch.cat([x_m, feat_mask.float()], dim=-1)
        h_m = self.proj(x_in)

        return h_m


# Attention-based Graph Constructor
class ModalityGraphConstructor(nn.Module):
    """Multi-head attention constructor for modality-specific graphs."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_att = nn.Linear(self.num_heads, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, h_m):
        """
        Compute modality-specific adjacency matrix A^m using multi-head attention.
        Q^m_i = W^{Q_m}_i · h^m, K^m_i = W^{K_m}_i · h^m
        A^m = [AttHead^m_1 ⊕ ... ⊕ AttHead^m_C] W_att
        """
        B, _ = h_m.shape
        Q_m = self.W_Q(h_m).view(B, self.num_heads, self.d_k)
        K_m = self.W_K(h_m).view(B, self.num_heads, self.d_k)

        # Attention for each head: AttHead^m_i = softmax(Q^m_i · (K^m_i)^T)
        heads = []
        for i in range(self.num_heads):
            Q_m_i, K_m_i = Q_m[:, i, :], K_m[:, i, :]
            scores = torch.mm(Q_m_i, K_m_i.T) / math.sqrt(self.d_k)
            scores = scores.masked_fill(torch.eye(B, device=scores.device, dtype=torch.bool), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.dropout:
                attn = self.dropout(attn)
            heads.append(attn)

        # Combine heads: A^m = [heads] W_att
        heads = torch.stack(heads, dim=-1)
        A_m = self.W_att(heads).squeeze(-1)
        A_m = torch.relu(A_m)  # Ensures non-negative

        return A_m


# Multi-Modal Graph (Attention)
class MultiModalGraphConstructor(nn.Module):
    """Combines modality-specific graphs"""
    def __init__(self, modality_dims, hidden_dim=128, num_heads=4, theta=0.5, top_k=None,
                 dropout=0.0, aggregate=True, filter_graph=True, reliable_ratio=True):
        super().__init__()
        self.theta = theta
        self.top_k = top_k
        self.aggregate = aggregate
        self.filter_graph_flag = filter_graph
        self.reliable_ratio_flag = reliable_ratio

        # Encoder and constructor for each modality
        self.encoders = nn.ModuleDict({m: ModalityEncoder(modality_dims[m], hidden_dim) for m in modality_dims})
        self.constructors = nn.ModuleDict({m: ModalityGraphConstructor(hidden_dim, num_heads, dropout) for m in modality_dims})

    def _filter_graph(self, A):
        """Top-k or threshold filtering"""
        if self.top_k:
            N = A.shape[0]
            k = min(self.top_k, N - 1)
            kth_vals = torch.topk(A, k, dim=1).values[:, -1]
            keep = A >= kth_vals.unsqueeze(1)
            A = A * keep.float()
        else:
            A = torch.where(A > self.theta, A, torch.zeros_like(A))

        # Row-normalize so each patient's top-k weights sum to 1.
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A = A / row_sum

        return A

    def forward(self, x_dict, feat_mask_dict, patient_mask_dict):
        """Build patient similarity graph from multi-modal data.
        A_hat = (Σ A^m · mask^m) / (Σ mask^m + ε)"""
        A_m_dict, validity_masks, h_m_dict = {}, {}, {}

        # Step 1: Encode each modality and compute modality-specific adjacency A^m
        for m in sorted(x_dict.keys()):
            x_m = x_dict[m]
            feat_mask = feat_mask_dict[m]

            # h^m = f^m_Φ(x^m) - Feature extraction
            encoder = self.encoders[m]
            h_m = encoder(x_m, feat_mask)

            # A^m - Modality-specific adjacency via multi-head attention
            constructor = self.constructors[m]
            A_m = constructor(h_m)

            # Step 2: Compute validity mask (mask^m)
            # Controls how much each patient pair contributes when aggregating modality graphs.
            patient_present = (feat_mask.sum(dim=1) > 0).float()   # [B], binary presence
            if self.reliable_ratio_flag:
                # reliable_ratio[i] = fraction of observed features for patient i in modality m
                # Pairwise weight = product of both ratios: 1.0 when both fully observed,
                # proportionally lower when either is sparse, 0 when either is completely missing
                D_m = x_m.shape[1]
                ratio = feat_mask.sum(dim=1) / D_m                  # [B], fractional completeness
                validity_mask = ratio.unsqueeze(1) * ratio.unsqueeze(0)
            else:
                # Binary: 1 if both patients have any data for this modality, 0 otherwise
                validity_mask = patient_present.unsqueeze(1) * patient_present.unsqueeze(0)

            # Store results
            h_m_dict[m] = h_m
            A_m_dict[m] = A_m
            validity_masks[m] = validity_mask

        # Step 3: Aggregate across modalities to get A_hat. A_hat = (Σ A^m · mask^m) / (Σ mask^m + ε)
        numerator = sum(A_m_dict[m] * validity_masks[m] for m in sorted(A_m_dict.keys()))
        denominator = sum(validity_masks[m] for m in sorted(validity_masks.keys()))

        # Handle pairs with no valid comparisons (denominator=0)
        valid_pairs = denominator > 1e-8
        A_hat = torch.zeros_like(numerator)
        A_hat[valid_pairs] = (numerator[valid_pairs] / denominator[valid_pairs])

        # Apply filtering (top-k or threshold θ) or plain row-normalization
        if self.filter_graph_flag:
            A_hat = self._filter_graph(A_hat)
        else:
            # Row-normalize without top-k filtering
            row_sum = A_hat.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A_hat = A_hat / row_sum

        modality_info = {
            **{f"h_m_{m}": h_m_dict[m] for m in sorted(h_m_dict.keys())},
            **{f"A_m_{m}": A_m_dict[m] for m in sorted(A_m_dict.keys())},
            **{f"{m}_validity": validity_masks[m] for m in sorted(validity_masks.keys())}
        }

        return A_hat, modality_info