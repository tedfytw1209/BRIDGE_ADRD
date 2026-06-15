import torch
import pandas as pd
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import config

ARTIFACTS_DIR = os.path.join("output", "artifacts")
SPLIT_PATH = os.path.join(ARTIFACTS_DIR, "default_split.npz")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "global_scalers.pkl")


def _get_paths(output_dir=None):
    """Artifact paths derived from output_dir."""
    if output_dir is None:
        output_dir = config.config.get('output_dir', 'output')
    artifacts_dir = os.path.join(output_dir, "artifacts")
    return {
        'artifacts_dir': artifacts_dir,
        'split_path': os.path.join(artifacts_dir, "default_split.npz"),
        'scaler_path': os.path.join(artifacts_dir, "global_scalers.pkl"),
        'clip_path': os.path.join(artifacts_dir, "clip_thresholds.pkl"),
        'mask_dir': os.path.join(output_dir, "eval_masks"),
    }


class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, modality_data, labels, patient_ids):
        self.modality_data = {key: torch.tensor(val, dtype=torch.float32) for key, val in modality_data.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.patient_ids = patient_ids
        self.modality_keys = sorted(modality_data.keys())

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {"label": self.labels[idx], "patient_id": self.patient_ids[idx]}
        # Add all modalities
        for key in self.modality_keys:
            item[key] = self.modality_data[key][idx]

        return item


def align_by_ids(df, base_ids, name, allow_missing=False):
    if "subid" not in df.columns: raise ValueError(f"{name}: 'subid' column is missing. Available: {list(df.columns)}")

    df = df.set_index("subid")
    missing = set(base_ids) - set(df.index.values)
    if missing and not allow_missing: raise ValueError(f"{name}: missing {len(missing)} subid(s)")
    return df.reindex(base_ids)


def load_multimodal_data(data_paths, base_ids):
    modalities_to_load = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
    modality_data = {}
    modality_columns = {}  # {mod_key: list of column names} before any encoding
    for mod_key in modalities_to_load:
        path_key = f'{mod_key}_features'
        file_path = data_paths[path_key]
        allow_missing = mod_key in ['m3', 'm5', 'm6', 'm7']

        df = pd.read_csv(file_path)
        aligned_df = align_by_ids(df, base_ids, name=mod_key.upper(), allow_missing=allow_missing)
        modality_columns[mod_key] = list(aligned_df.columns)
        modality_data[mod_key] = aligned_df.to_numpy(dtype=np.float32)

        if mod_key == 'm1': modality_data[mod_key] = clean_m1_race(modality_data[mod_key])
        if mod_key == 'm6': modality_data[mod_key] = clean_m6_apoe(modality_data[mod_key])

        n_features = modality_data[mod_key].shape[1]
        missing_pct = np.isnan(modality_data[mod_key]).sum() / modality_data[mod_key].size * 100
        print(f"  {mod_key.upper()}: {n_features} features, {missing_pct:.1f}% missing")

    return modality_data, modality_columns


def clean_m1_race(m1_data):
    race_col = m1_data[:, 2].copy()
    mode_value = stats.mode(race_col[~np.isnan(race_col)], keepdims=True)[0][0]
    race_col = np.where(np.isnan(race_col), mode_value, race_col)

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    race_encoded = enc.fit_transform(race_col.reshape(-1, 1))

    m1_encoded = np.concatenate([
        m1_data[:, :2],  # Binary features before RACE
        race_encoded,  # One-hot encoded RACE
        m1_data[:, 3:]  # Binary age bins
    ], axis=1)
    print(f"M1 shape after one-hot: {m1_encoded.shape[1]}")

    return m1_encoded


def clean_m6_apoe(m6_data):
    """One-hot encode APOEGenotype (categorical: 22, 23, 24, 33, 34, 44), preserving NaN."""
    apoe_col = m6_data[:, 0].copy()
    nan_mask = np.isnan(apoe_col)

    categories = [22.0, 23.0, 24.0, 33.0, 34.0, 44.0]
    enc = OneHotEncoder(sparse_output=False, categories=[categories], handle_unknown='ignore')

    # Fill NaN temporarily for encoding
    temp_col = apoe_col.copy()
    temp_col[nan_mask] = categories[0]
    encoded = enc.fit_transform(temp_col.reshape(-1, 1))

    # Restore NaN for missing patients
    encoded[nan_mask] = np.nan

    print(f"  M6 shape after one-hot: {encoded.shape[1]} (categories: {categories})")
    return encoded


def compute_clip_thresholds(train_data, modality_columns, outlier_clip_config):
    """Compute percentile thresholds from train data only."""
    clip_thresholds = {}
    for mod_key, clip_cfg in outlier_clip_config.items():
        if mod_key not in train_data:
            continue
        percentile = clip_cfg['percentile']
        features_to_clip = set(clip_cfg['features'])
        col_names = modality_columns.get(mod_key, [])
        thresholds = {}
        for j, col_name in enumerate(col_names):
            if col_name in features_to_clip:
                col_vals = train_data[mod_key][:, j]
                valid_vals = col_vals[~np.isnan(col_vals)]
                if len(valid_vals) > 0:
                    thresholds[j] = float(np.percentile(valid_vals, percentile))
                    print(f"  {mod_key}.{col_name} (col {j}): p{percentile} = {thresholds[j]:.4f}")
        clip_thresholds[mod_key] = thresholds
    return clip_thresholds


def apply_clip_thresholds(data, clip_thresholds):
    """Clip values using thresholds; NaNs preserved."""
    for mod_key, thresholds in clip_thresholds.items():
        if mod_key not in data:
            continue
        for j, thresh in thresholds.items():
            col = data[mod_key][:, j]
            n_clipped = int(np.nansum(col > thresh))
            col_clipped = np.where(np.isnan(col), col, np.minimum(col, thresh))
            data[mod_key][:, j] = col_clipped
            if n_clipped > 0:
                print(f"  {mod_key} col {j}: clipped {n_clipped} values to {thresh:.4f}")
    return data


CLIP_PATH = os.path.join(ARTIFACTS_DIR, "clip_thresholds.pkl")


def create_split(config, save=True):
    obs_periods = pd.read_csv(config["data_paths"]["obs_periods"])
    patient_ids = obs_periods["subid"].values
    label_mapping = {"Normal": 0, "MCI": 1, "Dementia": 2}
    mapped = obs_periods["label"].map(label_mapping)

    if mapped.isna().any():
        missing_labels = obs_periods.loc[mapped.isna(), "label"].unique()
        raise ValueError(f"Unmapped labels: {missing_labels}. Expected: {list(label_mapping.keys())}")

    labels = mapped.values.astype(np.int16)

    # Load all modalities & create train/test split
    modality_data, modality_columns = load_multimodal_data(config["data_paths"], base_ids=patient_ids)
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=config["test_size"], random_state=config["random_state"], stratify=labels)
    train_data = {"labels": labels[train_idx], "patient_ids": patient_ids[train_idx]}
    test_data = {"labels": labels[test_idx], "patient_ids": patient_ids[test_idx]}

    # Add modalities to train/test
    for mod_key, mod_array in modality_data.items():
        train_data[mod_key] = mod_array[train_idx]
        test_data[mod_key] = mod_array[test_idx]

    paths = _get_paths()
    outlier_clip_config = config.get('outlier_clip', {})
    if outlier_clip_config:
        print("\nComputing outlier clip thresholds from training data:")
        clip_thresholds = compute_clip_thresholds(train_data, modality_columns, outlier_clip_config)

        for mod_key in clip_thresholds:
            if mod_key in test_data:
                test_data[f'{mod_key}_original'] = test_data[mod_key].copy()
                train_data[f'{mod_key}_original'] = train_data[mod_key].copy()

        print("Clipping training data:")
        apply_clip_thresholds(train_data, clip_thresholds)
        print("Clipping test data (using train thresholds):")
        apply_clip_thresholds(test_data, clip_thresholds)

        # Save thresholds for later use
        if save:
            os.makedirs(paths['artifacts_dir'], exist_ok=True)
            joblib.dump(clip_thresholds, paths['clip_path'])
            print(f"Clip thresholds saved to {paths['clip_path']}")

    modality_dims = {mod_key: mod_array.shape[1] for mod_key, mod_array in modality_data.items()}

    if save:
        os.makedirs(paths['artifacts_dir'], exist_ok=True)
        np.savez_compressed(paths['split_path'], train_data=train_data, test_data=test_data, modality_dims=modality_dims)
        print(f"Split saved to {paths['split_path']}")

    return train_data, test_data, modality_dims


def load_split():
    paths = _get_paths()
    split_path = paths['split_path']
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split not found: {split_path}")
    data = np.load(split_path, allow_pickle=True)
    train_data = data["train_data"].item()
    test_data = data["test_data"].item()
    modality_dims = data["modality_dims"].item()
    return train_data, test_data, modality_dims


def load_clip_thresholds(path=None):
    if path is None:
        path = _get_paths()['clip_path']
    if not os.path.exists(path):
        return {}
    return joblib.load(path)


def get_or_create_split(config, save=True):
    paths = _get_paths()
    split_path = paths['split_path']
    if os.path.exists(split_path):
        print(f"Loading existing split from {split_path}")
        return load_split()
    return create_split(config, save=save)


def fit_global_scalers(modality_data, modalities_to_scale=None, save_path=None):
    if save_path is None:
        save_path = _get_paths()['scaler_path']
    modalities_to_scale = modalities_to_scale or config.config['modalities_to_scale']
    scalers = {}
    medians = {}

    for mod_key in modalities_to_scale:
        if mod_key not in modality_data:
            continue
        X = modality_data[mod_key]
        med = np.nanmedian(X, axis=0)
        medians[mod_key] = med
        scalers_mod = {}
        for j in range(X.shape[1]):
            if not np.isfinite(med[j]):
                scalers_mod[j] = None
                continue
            col = X[:, j]
            filled = np.where(np.isnan(col), med[j], col)
            scalers_mod[j] = StandardScaler().fit(filled.reshape(-1, 1))
        scalers[mod_key] = scalers_mod

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"scalers": scalers, "medians": medians}, save_path)
    print(f"Saved global scalers to {save_path}")
    return scalers, medians


def load_global_scalers(path=None):
    if path is None:
        path = _get_paths()['scaler_path']
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global scalers not found: {path}")
    data = joblib.load(path)
    return data["scalers"], data["medians"]


def apply_global_scalers(modality_data, scalers, medians, modalities_to_scale=None):
    modalities_to_scale = modalities_to_scale or config.config['modalities_to_scale']
    scaled = {}
    for mod_key, X in modality_data.items():
        if mod_key not in modalities_to_scale or mod_key not in scalers:
            scaled[mod_key] = X
            continue
        X_out = X.copy()
        nan_mask = np.isnan(X_out)
        med = medians.get(mod_key)
        for j, sc in scalers[mod_key].items():
            if sc is None:
                continue
            col = X_out[:, j]
            filled = np.where(np.isnan(col), med[j], col)
            X_out[:, j] = sc.transform(filled.reshape(-1, 1)).ravel()
        X_out[nan_mask] = np.nan
        scaled[mod_key] = X_out
    return scaled


def create_fold_dataloaders(modality_data, labels, ids, train_idx, val_idx, batch_size=64, num_workers=0, pin_memory=False):
    # Split data by fold indices
    modality_keys = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
    train_modalities = {}
    val_modalities = {}

    for mod_key in [k for k in modality_keys if k in modality_data]:
        train_modalities[mod_key] = modality_data[mod_key][train_idx].copy()
        val_modalities[mod_key] = modality_data[mod_key][val_idx].copy()

    # Create datasets & dataloaders
    train_ds = PatientDataset(train_modalities, labels[train_idx], ids[train_idx])
    val_ds = PatientDataset(val_modalities, labels[val_idx], ids[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_or_create_eval_mask(test_data, modality_key, mask_prob=0.3, seed=42, force_regenerate=False):
    mask_dir = _get_paths()['mask_dir']
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"R_eval_test_{modality_key}.npy")

    # Remove stale masks for this modality with different params
    import glob
    for old in glob.glob(os.path.join(mask_dir, f"R_eval_test_{modality_key}*.npy")):
        if old != mask_path:
            os.remove(old)
            print(f"Removed stale mask: {old}")

    # Load existing mask if available
    if os.path.exists(mask_path) and not force_regenerate:
        R_eval = np.load(mask_path)
        # Validate: R_eval should only mask observed (non-NaN) positions
        conflicts = (R_eval & np.isnan(test_data)).sum()
        if R_eval.shape == test_data.shape and conflicts == 0:
            print(f"\nLoaded existing R_eval mask for {modality_key.upper()} from {mask_path}")
            observed = ~np.isnan(test_data)
            hidden = R_eval.sum()
            pct = 100 * hidden / observed.sum() if observed.sum() > 0 else 0
            print(f"Hidden entries: {hidden} ({pct:.1f}% of observed values)")
            return R_eval
        else:
            reason = "shape mismatch" if R_eval.shape != test_data.shape else f"{conflicts} masked positions are NaN in data (old mask)"
            print(f"WARNING: {reason} for {modality_key.upper()}, regenerating...")

    # Generate new mask using a local RNG
    rng = np.random.RandomState(seed)
    present_feat = ~np.isnan(test_data)

    categorical_modalities = set(config.config['categorical_modalities'])
    if modality_key in categorical_modalities:
        # Row-wise masking for categorical: mask entire one-hot vector or nothing
        patient_has_data = np.any(present_feat, axis=1)
        patient_mask = (rng.rand(test_data.shape[0]) < mask_prob) & patient_has_data
        R_eval = np.zeros(test_data.shape, dtype=bool)
        R_eval[patient_mask] = True
    else:
        R_eval = (rng.rand(*test_data.shape) < mask_prob) & present_feat

    np.save(mask_path, R_eval)
    print(f"Created and saved R_eval mask for {modality_key.upper()} to {mask_path}")
    print(f"Hidden entries: {R_eval.sum()} ({100 * R_eval.sum() / present_feat.sum():.1f}% of observed values)")

    return R_eval
