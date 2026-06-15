"""
grape_imputer.py - PyTorch port of GRAPE (You et al., NeurIPS 2020).
Source: github.com/maxiaoba/GRAPE

Architecture faithful to paper:
  - 3-layer EGraphSage (edge_mode=1): message = ReLU(W · [x_j ‖ e_ij])
  - Edge-update MLPs after each GNN layer (ported from GNNStack)
  - Imputation MLP head: [x_src ‖ x_dst] → scalar (ported from MLPNet)
  - MinMax normalization fit on X_train, applied to X_test

Key adaptation vs original (transductive):
  fit(X_train)  trains on X_train's bipartite graph using random known-edge
                masking each epoch (known=0.7 → 70 % of edges visible to GNN, loss on all edges).
  transform(X)  builds a combined bipartite graph [X_train observed + X observed], runs frozen GNN, fills NaN positions in X via
                the imputation MLP.  X_train context improves feature-node embeddings for more accurate test-sample predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing



# EGraphSage (ported from models/egsage.py, edge_mode=1)
class _EGraphSage(MessagePassing):
    """Edge-aware GraphSAGE layer (edge_mode=1).

    message : ReLU( W_msg · [x_j ‖ e_ij] )
    update  : ReLU( W_agg · [aggr ‖ x_i] )  [+ optional L2 normalisation]
    """

    def __init__(self, in_channels, out_channels, edge_channels, normalize_emb=True, aggr='mean'):
        super().__init__(aggr=aggr)
        self.message_lin   = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin       = nn.Linear(in_channels + out_channels,  out_channels)
        self.act           = nn.ReLU()
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_j, edge_attr):
        return self.act(self.message_lin(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out, x):
        out = self.act(self.agg_lin(torch.cat([aggr_out, x], dim=-1)))
        if self.normalize_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out


# GNNStack (ported from models/gnn_model.py — EGSAGE variant)
class _GNNStack(nn.Module):
    """n_layers of EGraphSage with per-layer edge-feature update MLPs, followed by a post-node MLP."""

    def __init__(self, node_input_dim, edge_input_dim, node_dim, edge_dim, n_layers=3, dropout=0.0):
        super().__init__()
        # EGraphSage convolutions
        self.convs = nn.ModuleList([
            _EGraphSage(node_input_dim, node_dim, edge_input_dim),
            *[_EGraphSage(node_dim, node_dim, edge_dim) for _ in range(n_layers - 1)]
        ])
        # Edge update MLPs (one per layer), matching GNNStack.build_edge_update_mlps
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim * 2 + edge_input_dim, edge_dim), nn.ReLU()),
            *[nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, edge_dim), nn.ReLU()) for _ in range(n_layers - 1)]
        ])
        # Post-node MLP (build_node_post_mlp with post_hiddens=[node_dim])
        self.post_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x, edge_attr, edge_index):
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):
            x = conv(x, edge_attr, edge_index)
            x_i = x[edge_index[0]]
            x_j = x[edge_index[1]]
            edge_attr = edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return self.post_mlp(x)


# MLPNet (ported from models/prediction_model.py)
class _MLPNet(nn.Module):
    """Imputation head: concatenate endpoint embeddings → scalar prediction."""

    def __init__(self, node_dim, hidden_dims=(64,), dropout=0.0):
        super().__init__()
        layers, in_d = [], node_dim * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_src, x_dst):
        return self.net(torch.cat([x_src, x_dst], dim=-1))


class GRAPEImputer:
    """
    GRAPE imputation baseline.
    Architecture: 3-layer EGraphSage + edge-update MLPs + imputation MLP head.

    fit(X_train)   trains on X_train's bipartite graph.
    transform(X)   combines [X_train observed + X observed] into one bipartite
                   graph, runs frozen model, returns X with NaN positions filled.
    """

    def __init__(self, node_dim=64, edge_dim=64, n_layers=3, known=0.7, epochs=3000, lr=1e-3, dropout=0.0, seed=None):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.known = known
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.seed = seed

        self._norm_params = None   # (mins, rng) fit on X_train
        self._X_train_norm = None   # normalised X_train (NaN preserved)
        self._N_train = None
        self._D = None
        self._gnn = None
        self._mlp = None

    # MinMax normalisation
    def _fit_normalize(self, X):
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        rng  = np.where(maxs > mins, maxs - mins, 1.0)
        self._norm_params = (mins, rng)
        return (X - mins) / rng

    def _normalize(self, X):
        mins, rng = self._norm_params
        return (X - mins) / rng

    def _renormalize(self, X_n):
        mins, rng = self._norm_params
        return X_n * rng + mins

    # bipartite graph builder
    def _build_graph(self, X_list):
        """
        Build a bidirectional bipartite graph from one or more data blocks that share the same D feature columns.
        Node layout:
          [0 .. N_total-1]          sample nodes  (rows of all X blocks)
          [N_total .. N_total+D-1]  feature nodes (shared across all blocks)

        Node features (mode-0 from uci/uci_data.py):
          sample nodes : all-ones vector of length D
          feature nodes : one-hot identity matrix (D × D)

        Edge attributes: normalised observed value (scalar, shape (2E, 1)).
        First E edges are forward (sample→feature); second E are reverse.

        Returns
        -------
        x : (N_total + D, D) float tensor
        edge_index : (2, 2E) long tensor
        edge_attr : (2E, 1) float tensor
        sample_offsets: list[int], start index of each block in node list
        E_half : int, number of forward edges
        """
        D = X_list[0].shape[1]
        N_total = sum(X.shape[0] for X in X_list)
        feat_offset = N_total

        # Node features
        sample_x = np.ones((N_total, D), dtype=np.float32)
        feat_x = np.eye(D, dtype=np.float32)
        x = torch.tensor(np.vstack([sample_x, feat_x]), dtype=torch.float)

        # Collect edges (vectorised per block)
        src_fwd_all, dst_fwd_all, vals_all = [], [], []
        offset = 0
        sample_offsets = []
        for X in X_list:
            sample_offsets.append(offset)
            row_idx, col_idx = np.where(~np.isnan(X))
            src_fwd_all.append(offset + row_idx)
            dst_fwd_all.append(feat_offset + col_idx)
            vals_all.append(X[row_idx, col_idx])
            offset += X.shape[0]

        src_fwd = np.concatenate(src_fwd_all)
        dst_fwd = np.concatenate(dst_fwd_all)
        vals = np.concatenate(vals_all)
        E_half = len(src_fwd)

        # Bidirectional
        src = np.concatenate([src_fwd, dst_fwd])
        dst = np.concatenate([dst_fwd, src_fwd])
        val = np.concatenate([vals,    vals])

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(val[:, None].astype(np.float32), dtype=torch.float)

        return x, edge_index, edge_attr, sample_offsets, E_half

    def fit(self, X_train):
        """Train GRAPE on X_train.
        X_train : (N, D) float array, NaN = missing.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        X_n = self._fit_normalize(X_train.copy().astype(float))
        self._X_train_norm = X_n
        N, D = X_n.shape
        self._N_train = N
        self._D = D

        x, edge_index, edge_attr, _, E_half = self._build_graph([X_n])
        train_labels = edge_attr[:E_half, 0].clone()   # ground-truth forward edges

        # Build models (node_input_dim = D, edge_input_dim = 1)
        self._gnn = _GNNStack(D, 1, self.node_dim, self.edge_dim, self.n_layers, self.dropout)
        self._mlp = _MLPNet(self.node_dim, hidden_dims=(self.edge_dim,), dropout=self.dropout)

        opt = torch.optim.Adam(list(self._gnn.parameters()) + list(self._mlp.parameters()), lr=self.lr)

        self._gnn.train()
        self._mlp.train()

        for epoch in range(self.epochs):
            # Random known mask - reveal `known` fraction of forward edges to GNN (mirrors get_known_mask + mask_edge in utils/utils.py)
            known_mask = torch.FloatTensor(E_half).uniform_() < self.known
            double_mask = torch.cat([known_mask, known_mask])
            known_ei = edge_index[:, double_mask]
            known_ea = edge_attr[double_mask]

            opt.zero_grad()
            x_emb = self._gnn(x, known_ea, known_ei)

            # Predict ALL forward edges (loss_mode=0, same as GRAPE default)
            src_all = edge_index[0, :E_half]
            dst_all = edge_index[1, :E_half]
            pred = self._mlp(x_emb[src_all], x_emb[dst_all]).squeeze(-1)

            loss = F.mse_loss(pred, train_labels)
            loss.backward()
            opt.step()

            if (epoch + 1) % 500 == 0:
                print(f"    epoch {epoch+1}/{self.epochs}  loss={loss.item():.4f}")

        self._gnn.eval()
        self._mlp.eval()
        return self

    # transform
    def transform(self, X):
        """Impute NaN positions in X.

        Builds a combined bipartite graph [X_train observed + X observed],
        runs frozen GNN, predicts each NaN position via the imputation MLP,
        and denormalises back to the original scale.

        X : (N_test, D) float array, NaN = missing or artificially masked.
        Returns : (N_test, D) float array with NaN positions filled.
        """
        assert self._gnn is not None, "Call fit() before transform()."

        X_n = self._normalize(X.copy().astype(float))
        X_n = np.clip(X_n, 0.0, 1.0)   # clip OOD test values into training range

        # Combined graph: training samples + test samples share feature nodes
        x, edge_index, edge_attr, offsets, _ = self._build_graph([self._X_train_norm, X_n])

        test_offset = offsets[1]                       # first test sample-node index
        feat_offset = self._N_train + X.shape[0]       # feature nodes start here

        with torch.no_grad():
            x_emb = self._gnn(x, edge_attr, edge_index)

            X_n_out = X_n.copy()   # normalized, NaN preserved at missing positions
            nan_rows, nan_cols = np.where(np.isnan(X))
            for i, j in zip(nan_rows, nan_cols):
                abs_i = test_offset + i
                abs_j = feat_offset + j
                X_n_out[i, j] = self._mlp(
                    x_emb[[abs_i]], x_emb[[abs_j]]).item()

        return self._renormalize(X_n_out)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
