import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PatientGCNEncoder(nn.Module):
    """GCN encoder for patient representations.
    h_hat^m = GCN(h^m, A_hat)"""

    def __init__(self, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, normalize=False)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h_m_dict, A_hat, patient_mask_dict=None):
        # Precompute shared edges for the case with no per-modality masking
        shared_edge_index = torch.nonzero(A_hat, as_tuple=False).T
        shared_edge_weight = A_hat[shared_edge_index[0], shared_edge_index[1]]

        # Apply GCN to each modality
        h_hat_m_dict = {}
        for m, h_m in h_m_dict.items():
            if patient_mask_dict is not None and m in patient_mask_dict:
                # Mask out patients missing modality m from being neighbors (column mask)
                neighbor_mask = patient_mask_dict[m].float()  # [N]
                A_m = A_hat * neighbor_mask.unsqueeze(0)      # zero columns of absent patients
                edge_index = torch.nonzero(A_m, as_tuple=False).T
                edge_weight = A_m[edge_index[0], edge_index[1]]
            else:
                edge_index = shared_edge_index
                edge_weight = shared_edge_weight

            h = self.dropout(h_m)
            for layer in self.layers:
                h = layer(h, edge_index, edge_weight)
                h = F.relu(h)
            h_hat_m_dict[m] = h

        return h_hat_m_dict