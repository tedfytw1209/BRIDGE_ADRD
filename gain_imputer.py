"""
PyTorch implementation of GAIN: Missing Data Imputation using Generative Adversarial Nets (Yoon et al., ICML 2018).
Original paper code: https://github.com/jsyoon0823/GAIN

Preserved from the paper:
    - Generator: 3-layer MLP with ReLU + Sigmoid output
    - Discriminator: 3-layer MLP with ReLU + Sigmoid output
    - Xavier weight initialization
    - MinMax normalization before training, reversed on output
    - Hint mechanism: H = M * Bernoulli(hint_rate)
    - Missing positions filled with uniform noise (0, 0.01) during training
    - Loss: D_loss = cross-entropy on real vs imputed per feature
            G_loss = adversarial + alpha * MSE on observed positions
    - Adam optimizer for both G and D
    - Default parameters: batch_size=128, hint_rate=0.9, alpha=100, iterations=10000

Changed vs the original:
    - TF1 -> PyTorch
    - fit(X_train) + transform(X_test) instead of gain(all_data)
        This lets us train on the training set and impute a masked test set.
    - Normalization params (min/max per feature) are fit on X_train
        and reused when transforming X_test, preventing data leakage.
    - During transform(), missing positions are filled with uniform(0, 0.01) noise,
        matching the original inference pass exactly.
"""

import numpy as np
import torch
import torch.nn as nn


class GAINImputer:

    def __init__(self, batch_size=128, hint_rate=0.9, alpha=100, iterations=10000, seed=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.norm_params = None
        self.generator = None
        self.discriminator = None
        self.dim = None

    # Normalization (same as utils.py in original repo)
    def _normalize(self, data, params=None):
        """MinMax normalize to [0, 1] range, ignoring NaNs when fitting."""
        norm_data = data.copy().astype(float)
        _, dim = norm_data.shape

        if params is None:
            min_val = np.zeros(dim)
            max_val = np.zeros(dim)
            for i in range(dim):
                min_val[i] = np.nanmin(norm_data[:, i])
                norm_data[:, i] -= min_val[i]
                max_val[i] = np.nanmax(norm_data[:, i])
                norm_data[:, i] /= (max_val[i] + 1e-6)
            params = {'min_val': min_val, 'max_val': max_val}
        else:
            for i in range(dim):
                norm_data[:, i] = (norm_data[:, i] - params['min_val'][i]) / (params['max_val'][i] + 1e-6)

        return norm_data, params

    def _renormalize(self, norm_data, params):
        """Reverse MinMax normalization back to original scale."""
        data = norm_data.copy()
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] * (params['max_val'][i] + 1e-6) + params['min_val'][i]
        return data

    # Network builder (same architecture as original: 3-layer MLP)
    def _build_net(self, in_dim, h_dim, out_dim):
        """
        Original paper: 3 fully connected layers, ReLU hidden, Sigmoid output.
        Input size is 2*dim because data and mask (or hint) are concatenated.
        Xavier init matches the original TF xavier_init() utility.
        """
        net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim), nn.Sigmoid()
        )
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        return net

    # ------------------------------------------------------------------
    # Training (fit)
    # ------------------------------------------------------------------

    def fit(self, X_train):
        """
        Train GAIN on X_train. NaN values are treated as naturally missing.

        In the original code, gain() takes all data at once and imputes it.
        Here we separate training so the model can later be applied to X_test.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        no, dim = X_train.shape
        self.dim = dim

        # --- Same normalization as original utils.normalization() ---
        norm_data, self.norm_params = self._normalize(X_train)
        norm_data_x = np.nan_to_num(norm_data, nan=0.0)
        # M: mask matrix 1 = observed, 0 = missing (same convention as paper)
        data_m = (~np.isnan(X_train)).astype(float)

        # --- Build G and D (input is 2*dim: data + mask concatenated) ---
        self.generator = self._build_net(dim * 2, dim, dim)
        self.discriminator = self._build_net(dim * 2, dim, dim)

        opt_G = torch.optim.Adam(self.generator.parameters())
        opt_D = torch.optim.Adam(self.discriminator.parameters())

        X_full = torch.FloatTensor(norm_data_x)
        M_full = torch.FloatTensor(data_m)

        self.generator.train()
        self.discriminator.train()

        # --- Training loop (mirrors the original iteration loop exactly) ---
        for it in range(self.iterations):

            # Sample mini-batch
            idx = np.random.permutation(no)[:self.batch_size]
            X_mb = X_full[idx].clone()
            M_mb = M_full[idx]

            # Uniform noise replaces missing values in the generator input
            # (original: Z_mb = uniform_sampler(0, 0.01, batch_size, dim))
            Z_mb = torch.FloatTensor(np.random.uniform(0, 0.01, (len(idx), dim)))

            # Hint vector: partially reveal the mask to the discriminator
            # (original: H_mb = M_mb * binary_sampler(hint_rate, batch_size, dim))
            H_mb = M_mb * torch.FloatTensor(np.random.binomial(1, self.hint_rate, (len(idx), dim)))

            # Fill missing positions with noise (observed stay as-is)
            X_mb_in = M_mb * X_mb + (1 - M_mb) * Z_mb

            # Generator forward pass: G(corrupted_data, mask)
            G_sample = self.generator(torch.cat([X_mb_in, M_mb], dim=1))

            # Combine: keep observed values, fill missing with G output
            # (original: Hat_X = X * M + G_sample * (1-M))
            Hat_X = X_mb_in * M_mb + G_sample * (1 - M_mb)

            # Discriminator step
            # D sees (imputed_data, hint) and predicts real (M=1) vs generated (M=0) per feature
            D_prob = self.discriminator(torch.cat([Hat_X.detach(), H_mb], dim=1))
            D_loss = -torch.mean(
                M_mb * torch.log(D_prob + 1e-8) +
                (1 - M_mb) * torch.log(1 - D_prob + 1e-8)
            )
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # Generator step
            # G_adv: fool D into thinking imputed values are real
            # MSE: reconstructed observed values should match actual observed values
            D_prob2 = self.discriminator(torch.cat([Hat_X, H_mb], dim=1))
            G_adv = -torch.mean((1 - M_mb) * torch.log(D_prob2 + 1e-8))
            MSE = torch.mean((M_mb * X_mb_in - M_mb * G_sample) ** 2) / (torch.mean(M_mb) + 1e-8)
            G_loss = G_adv + self.alpha * MSE
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            if (it + 1) % 2000 == 0:
                print(f"    iter {it+1}/{self.iterations}  D={D_loss.item():.4f}  G={G_loss.item():.4f}")

        self.generator.eval()
        self.discriminator.eval()
        return self

    # ------------------------------------------------------------------
    # Inference (transform)
    # ------------------------------------------------------------------

    def transform(self, X):
        """
        Impute missing values in X using the trained generator.

        In the original code, imputation is done on the same data used for
        training. Here we apply it to a new (possibly masked) test set.
        Normalization params from fit() are reused to avoid data leakage.

        Args:
            X: numpy array [N, D], NaN where missing (including R_eval positions)
        Returns:
            imputed: numpy array [N, D], no NaNs, in the same scale as X
        """
        if self.generator is None:
            raise RuntimeError("Call fit() before transform()")

        data_m = (~np.isnan(X)).astype(float)
        norm_data, _ = self._normalize(X, self.norm_params)
        norm_data_x = np.nan_to_num(norm_data, nan=0.0)

        # Match original: fill missing positions with small uniform noise before
        # passing to the generator (same as training, avoids train-test mismatch)
        Z_mb = np.random.uniform(0, 0.01, X.shape)
        X_in = data_m * norm_data_x + (1 - data_m) * Z_mb

        X_tensor = torch.FloatTensor(X_in)
        M_tensor = torch.FloatTensor(data_m)

        with torch.no_grad():
            G_sample = self.generator(torch.cat([X_tensor, M_tensor], dim=1)).numpy()

        # Keep observed positions, fill missing with generator output
        imputed_norm = data_m * norm_data_x + (1 - data_m) * G_sample
        return self._renormalize(imputed_norm, self.norm_params)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
