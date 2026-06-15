import os
import random
import numpy as np
import torch
from data_loader import load_global_scalers


def set_deterministic_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ← add this first
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# Metrics
def mae(a, b): return float(np.nanmean(np.abs(a - b)))
def mse(a, b): return float(np.nanmean((a - b) ** 2))
def rmse(a, b): return float(np.sqrt(np.nanmean((a - b) ** 2)))

def mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & (np.abs(y_true) > epsilon)
    if not np.any(mask): return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.clip(np.abs(y_true[mask]), epsilon, None))) * 100)

def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.nanmean(y_true))**2) + 1e-12
    return float(1 - ss_res / ss_tot)

def compute_all_metrics(y_true, y_pred):
    return {"mae": mae(y_true, y_pred), "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),}

def to_onehot(continuous_preds):
    """Convert continuous predictions to one-hot via argmax."""
    pred_class = np.argmax(continuous_preds, axis=1)
    onehot = np.zeros_like(continuous_preds)
    onehot[np.arange(len(pred_class)), pred_class] = 1.0
    return onehot


def transform_modality(scalers_dict, medians_dict, X, mod_key):
    if mod_key not in scalers_dict or scalers_dict[mod_key] is None: return X.copy()
    X_out = X.copy()
    nan_mask = np.isnan(X_out)
    med = medians_dict.get(mod_key)
    for j, sc in scalers_dict[mod_key].items():
        if sc is None: continue
        col = X_out[:, j]
        filled = np.where(np.isnan(col), med[j], col)
        X_out[:, j] = sc.transform(filled.reshape(-1, 1)).ravel()
    X_out[nan_mask] = np.nan
    return X_out

def inverse_transform_modality(scalers_dict, X_scaled, mod_key):
    if mod_key not in scalers_dict or scalers_dict[mod_key] is None: return X_scaled.copy()
    X_orig = np.full_like(X_scaled, np.nan)
    for j, sc in scalers_dict[mod_key].items():
        if sc is not None:
            col = X_scaled[:, j]
            mask = ~np.isnan(col)
            if mask.any():
                X_orig[mask, j] = sc.inverse_transform(col[mask].reshape(-1, 1)).ravel()
    return X_orig

def round_m5_features(m5_data):
    """Clip/round M5 features to valid ranges after bounded normalization.
    7 features (MMSE, MOCATOTS, CDRSUM, MINTTOTS, CRAFTURS, REY1REC, MOCA_CLOCK_TOTAL) are normalized to [0, 1] - just clip to that range.
    CDRGLOB (index 3) and GDS (index 7) are ordinal integers, so round then clip.
    """
    m5_data = np.asarray(m5_data, dtype=np.float32)
    if m5_data.ndim != 2:
        return m5_data
    m5_data = m5_data.copy()

    # Bounded features normalized to [0, 1]: clip to valid range
    for idx in [0, 1, 2, 4, 5, 6, 8]:   # MMSE, MOCATOTS, CDRSUM, MINTTOTS, CRAFTURS, REY1REC, MOCA_CLOCK_TOTAL
        m5_data[:, idx] = np.clip(m5_data[:, idx], 0.0, 1.0)

    # CDRGLOB (index 3): ordinal {0, 1, 2}
    m5_data[:, 3] = np.clip(np.round(m5_data[:, 3]), 0, 2)

    # GDS (index 7): ordinal {0, 1, 2, 3}
    m5_data[:, 7] = np.clip(np.round(m5_data[:, 7]), 0, 3)

    return m5_data