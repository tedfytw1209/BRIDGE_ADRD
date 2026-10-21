import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (exposes IterativeImputer)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

from data_loader import get_or_create_split, PatientDataset, get_or_create_eval_mask, load_global_scalers
from graph_builder import MultiModalGraphConstructor
from gcn import PatientGCNEncoder
from fusion import ModalityFusion
from decoders import ModalityDecoder
import config
from utils import transform_modality, inverse_transform_modality, to_onehot, round_m5_features, set_deterministic_seed
from evaluation import evaluate_categorical, evaluate_continuous, evaluate_continuous_trimmed, save_results_csv, plot_bars_by_feature, plot_single_feature, plot_categorical_metrics
from gain_imputer import GAINImputer
from grape_imputer import GRAPEImputer


def parse_and_override(cfg):
    """Parse CLI args and override cfg in-place. Returns parsed args."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',      type=str,   default=None)
    p.add_argument('--random-state',    type=int,   default=None)
    p.add_argument('--skip-graph',      type=int,   default=None)
    p.add_argument('--filter-graph',    type=int,   default=None)
    p.add_argument('--fusion-mode',     type=str,   default=None)
    p.add_argument('--lambda-cl',       type=float, default=None)
    p.add_argument('--reliable-ratio',  type=int,   default=None)
    p.add_argument('--top-k',           type=int,   default=None)
    p.add_argument('--num-gcn-layers',  type=int,   default=None)
    p.add_argument('--wandb-run-name',  type=str,   default=None)
    p.add_argument('--modality-a',      type=int,   default=None)
    args, _ = p.parse_known_args()

    if args.output_dir     is not None: cfg['output_dir']     = args.output_dir
    if args.random_state   is not None: cfg['random_state']   = args.random_state
    if args.skip_graph     is not None: cfg['skip_graph']     = args.skip_graph
    if args.filter_graph   is not None: cfg['filter_graph']   = args.filter_graph
    if args.fusion_mode    is not None: cfg['fusion_mode']    = args.fusion_mode
    if args.lambda_cl      is not None: cfg['lambda_cl']      = args.lambda_cl
    if args.reliable_ratio is not None: cfg['reliable_ratio'] = args.reliable_ratio
    if args.top_k          is not None: cfg['top_k']          = args.top_k
    if args.num_gcn_layers is not None: cfg['num_gcn_layers'] = args.num_gcn_layers
    if args.modality_a     is not None: cfg['modality_a']     = args.modality_a

    out = cfg['output_dir']
    cfg['save_model_path']     = os.path.join(out, 'output', 'best_model.pth')
    cfg['finetune_model_path'] = os.path.join(out, 'output', 'best_model_finetuned.pth')

    return args


@torch.no_grad()
def model_impute(loader, graph_builder, fusion, gcn_encoder, decoders, R_eval_dict, device, apply_mask=True):
    graph_builder.eval(); fusion.eval(); gcn_encoder.eval()
    for d in decoders.values(): d.eval()

    results = {}
    for mod_key in decoders.keys():
        results[mod_key] = {'true_std': [], 'pred_std': [], 'pred_scores': [], 'patient_ids': []}

    all_embeddings = []
    all_patient_ids = []

    categorical_mods = set(config.config['categorical_modalities'])
    ptr = 0
    for batch in loader:
        B = batch[list(batch.keys())[0]].shape[0]
        sl = slice(ptr, ptr + B)
        ptr += B
        pids = batch["patient_id"].numpy() if not torch.is_tensor(batch["patient_id"]) else batch["patient_id"].cpu().numpy()
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        available_modalities = [f'm{i}' for i in range(1, 8) if f'm{i}' in batch]
        x_dict = {m: batch[m].clone() for m in available_modalities}
        feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in x_dict}
        x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in x_dict}
        patient_mask_dict = {m: torch.isfinite(batch[m]).any(dim=1).float() for m in x_dict}

        true_dict = {m: x_dict[m].clone() for m in decoders.keys() if m in x_dict}

        if apply_mask and R_eval_dict is not None:
            for mod_key in decoders.keys():
                if mod_key in R_eval_dict and mod_key in x_dict_filled:
                    r_slice = torch.from_numpy(R_eval_dict[mod_key][sl]).to(device)
                    x_dict_filled[mod_key][r_slice] = 0.0
                    feat_mask_dict[mod_key][r_slice] = 0.0
            patient_mask_dict = {m: (feat_mask_dict[m].sum(dim=1) > 0).float() for m in x_dict}

        # Random feature masking to prevent data leakage (no edge masking — not augmentation).
        # Ensures h_fused[m] is built without seeing the features the decoder will predict,
        # consistent with how LR/KNN/GRAPE build their predictions from partial observations.
        if not apply_mask:
            for m in sorted(x_dict_filled.keys()):
                prob = config.config.get(f"prob_{m}", 0.1)
                feat_mask = feat_mask_dict[m]
                if m in categorical_mods:
                    patient_keep = torch.bernoulli(
                        torch.ones(x_dict_filled[m].size(0), 1, device=device) * (1 - prob)
                    )
                    keep = patient_keep.expand_as(x_dict_filled[m])
                else:
                    keep = torch.bernoulli(torch.ones_like(x_dict_filled[m]) * (1 - prob))
                x_dict_filled[m] = x_dict_filled[m] * keep
                feat_mask_dict[m] = feat_mask * keep
            patient_mask_dict = {m: (feat_mask_dict[m].sum(dim=1) > 0).float() for m in x_dict}

        A_hat, modality_info = graph_builder(x_dict_filled, feat_mask_dict, patient_mask_dict)
        h_m_dict = {m: modality_info[f"h_m_{m}"] for m in x_dict}
        if config.config["skip_graph"]:
            h_hat_m_dict = h_m_dict
        else:
            _pmask = patient_mask_dict if config.config["modality_a"] else None
            h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)
        h_fused_dict = fusion(h_m_dict, h_hat_m_dict, patient_mask_dict)

        # build z, zero-pad absent modalities
        all_modalities = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
        B_size = list(h_fused_dict.values())[0].shape[0]
        hidden_dim = list(h_fused_dict.values())[0].shape[1]
        batch_embeddings = []
        for m in all_modalities:
            if m in h_fused_dict:
                batch_embeddings.append(h_fused_dict[m])
            else:
                batch_embeddings.append(torch.zeros(B_size, hidden_dim, device=device))

        # z shape: [B, 7 * hidden_dim]
        z = torch.cat(batch_embeddings, dim=1)
        all_embeddings.append(z.cpu().numpy())
        all_patient_ids.append(pids)

        x_hat_dict = {}
        for mod_key in decoders.keys():
            if mod_key in h_fused_dict:
                raw_pred = decoders[mod_key](h_fused_dict[mod_key], x_dict_filled[mod_key], feat_mask_dict[mod_key])
                # categorical: softmax scores for AUROC, argmax → one-hot
                if mod_key in categorical_mods:
                    results[mod_key]['pred_scores'].append(torch.softmax(raw_pred, dim=-1).cpu().numpy())
                    pred_class = raw_pred.argmax(dim=-1)
                    x_hat_dict[mod_key] = torch.zeros_like(raw_pred).scatter_(1, pred_class.unsqueeze(-1), 1.0)
                else:
                    x_hat_dict[mod_key] = raw_pred

        for mod_key in decoders.keys():
            if mod_key in true_dict and mod_key in x_hat_dict:
                results[mod_key]['true_std'].append(true_dict[mod_key].cpu().numpy())
                results[mod_key]['pred_std'].append(x_hat_dict[mod_key].cpu().numpy())
                results[mod_key]['patient_ids'].append(pids)

    for mod_key in results.keys():
        if len(results[mod_key]['true_std']) > 0:
            results[mod_key]['true_std'] = np.concatenate(results[mod_key]['true_std'], axis=0)
            results[mod_key]['pred_std'] = np.concatenate(results[mod_key]['pred_std'], axis=0)
            if len(results[mod_key]['pred_scores']) > 0:
                results[mod_key]['pred_scores'] = np.concatenate(results[mod_key]['pred_scores'], axis=0)
            results[mod_key]['patient_ids'] = np.concatenate(results[mod_key]['patient_ids'], axis=0)

    if len(all_embeddings) > 0:
        embeddings = np.concatenate(all_embeddings, axis=0)
        patient_ids = np.concatenate(all_patient_ids, axis=0)
    else:
        embeddings = np.array([])
        patient_ids = np.array([])

    results['embeddings'] = embeddings
    results['embedding_patient_ids'] = patient_ids

    return results


@torch.no_grad()
def model_impute_single(train_modality_data, test_modality_data, test_patient_ids, graph_builder, fusion, gcn_encoder, decoders, R_eval_dict, device, apply_mask=True):
    """For each test patient: combines all training patients + 1 test patient,
    run a forward pass, and extract the test patient's embedding and predictions."""
    graph_builder.eval(); fusion.eval(); gcn_encoder.eval()
    for d in decoders.values(): d.eval()

    results = {}
    for mod_key in decoders.keys():
        results[mod_key] = {'true_std': [], 'pred_std': [], 'pred_scores': [], 'patient_ids': []}

    all_embeddings = []
    all_patient_ids = []
    categorical_mods = set(config.config['categorical_modalities'])

    available_modalities = sorted([f'm{i}' for i in range(1, 8) if f'm{i}' in train_modality_data])
    n_train = train_modality_data[available_modalities[0]].shape[0]
    n_test = len(test_patient_ids)

    train_tensors = {m: torch.tensor(train_modality_data[m], dtype=torch.float32) for m in available_modalities}
    train_feat_masks = {m: torch.isfinite(train_tensors[m]).float() for m in available_modalities}
    train_filled = {m: torch.nan_to_num(train_tensors[m], nan=0.0).to(device) for m in available_modalities}
    train_feat_masks_dev = {m: train_feat_masks[m].to(device) for m in available_modalities}
    train_patient_masks_dev = {m: train_feat_masks[m].any(dim=1).float().to(device) for m in available_modalities}

    print(f"  Processing {n_test} test patients individually (each with {n_train} training patients)")

    for i in range(n_test):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Patient {i+1}/{n_test}")

        test_row = {m: torch.tensor(test_modality_data[m][i:i+1], dtype=torch.float32) for m in available_modalities}
        test_feat_mask_i = {m: torch.isfinite(test_row[m]).float() for m in available_modalities}
        test_filled_i = {m: torch.nan_to_num(test_row[m], nan=0.0) for m in available_modalities}
        test_patient_mask_i = {m: test_feat_mask_i[m].any(dim=1).float() for m in available_modalities}

        true_dict_i = {m: test_row[m].clone() for m in decoders.keys() if m in test_row}

        if apply_mask and R_eval_dict is not None:
            for mod_key in decoders.keys():
                if mod_key in R_eval_dict and mod_key in test_filled_i:
                    r_mask = torch.from_numpy(R_eval_dict[mod_key][i:i+1])
                    test_filled_i[mod_key][r_mask] = 0.0
                    test_feat_mask_i[mod_key][r_mask] = 0.0
            test_patient_mask_i = {m: test_feat_mask_i[m].any(dim=1).float() for m in available_modalities}

        # Combine: all training patients + 1 test patient
        combined_filled = {m: torch.cat([train_filled[m], test_filled_i[m].to(device)], dim=0) for m in available_modalities}
        combined_feat_masks = {m: torch.cat([train_feat_masks_dev[m], test_feat_mask_i[m].to(device)], dim=0) for m in available_modalities}
        combined_patient_masks = {m: torch.cat([train_patient_masks_dev[m], test_patient_mask_i[m].to(device)], dim=0) for m in available_modalities}

        A_hat, modality_info = graph_builder(combined_filled, combined_feat_masks, combined_patient_masks)
        h_m_dict = {m: modality_info[f"h_m_{m}"] for m in available_modalities}
        if config.config["skip_graph"]:
            h_hat_m_dict = h_m_dict
        else:
            _pmask = combined_patient_masks if config.config["modality_a"] else None
            h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)
        h_fused_dict = fusion(h_m_dict, h_hat_m_dict, combined_patient_masks)

        # Extract test patient (last row, index = n_train)
        ti = n_train
        all_mods = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
        hidden_dim = list(h_fused_dict.values())[0].shape[1]
        emb_parts = []
        for m in all_mods:
            if m in h_fused_dict:
                emb_parts.append(h_fused_dict[m][ti:ti+1])
            else:
                emb_parts.append(torch.zeros(1, hidden_dim, device=device))
        z_i = torch.cat(emb_parts, dim=1)
        all_embeddings.append(z_i.cpu().numpy())
        all_patient_ids.append(test_patient_ids[i])

        for mod_key in decoders.keys():
            if mod_key in h_fused_dict:
                pred = decoders[mod_key](
                    h_fused_dict[mod_key][ti:ti+1],
                    combined_filled[mod_key][ti:ti+1],
                    combined_feat_masks[mod_key][ti:ti+1],
                )
                if mod_key in categorical_mods:
                    results[mod_key]['pred_scores'].append(torch.softmax(pred, dim=-1).cpu().numpy())
                    pred_class = pred.argmax(dim=-1)
                    pred = torch.zeros_like(pred).scatter_(1, pred_class.unsqueeze(-1), 1.0)
                results[mod_key]['true_std'].append(true_dict_i[mod_key].numpy())
                results[mod_key]['pred_std'].append(pred.cpu().numpy())
                results[mod_key]['patient_ids'].append(test_patient_ids[i:i+1])

    for mod_key in results.keys():
        if len(results[mod_key]['true_std']) > 0:
            results[mod_key]['true_std'] = np.concatenate(results[mod_key]['true_std'], axis=0)
            results[mod_key]['pred_std'] = np.concatenate(results[mod_key]['pred_std'], axis=0)
            if len(results[mod_key]['pred_scores']) > 0:
                results[mod_key]['pred_scores'] = np.concatenate(results[mod_key]['pred_scores'], axis=0)
            results[mod_key]['patient_ids'] = np.concatenate(results[mod_key]['patient_ids'], axis=0)

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
    patient_ids = np.array(all_patient_ids)
    results['embeddings'] = embeddings
    results['embedding_patient_ids'] = patient_ids

    return results


def mice_impute(X_train, X_target, n_imputations=5, max_iter=10, seed=42, extra_train=None, extra_target=None):
    """MICE (van Buuren 2011; Zhang 2016) via sklearn IterativeImputer.
    Draws n_imputations datasets with BayesianRidge + posterior sampling and pools them
    by averaging (Rubin's rule for the point estimate). extra_* are fully observed
    covariates appended as predictors only (used by PPM); they are not returned."""
    n_features = X_train.shape[1]
    if extra_train is not None:
        X_train = np.hstack([X_train, extra_train])
        X_target = np.hstack([X_target, extra_target])
    draws = []
    for k in range(n_imputations):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=max_iter,
                               initial_strategy="mean", imputation_order="ascending",
                               keep_empty_features=True, random_state=seed + k)
        imp.fit(X_train)
        draws.append(imp.transform(X_target)[:, :n_features])
    return np.mean(draws, axis=0)


def missforest_impute(X_train, X_target, n_estimators=100, max_iter=10, seed=42):
    """MissForest (Stekhoven & Buhlmann 2012) via sklearn IterativeImputer with a random forest:
    mean initial guess, columns visited in ascending missingness, mtry = sqrt(p)."""
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt", n_jobs=-1, random_state=seed)
    imp = IterativeImputer(estimator=rf, max_iter=max_iter, initial_strategy="mean",
                           imputation_order="ascending", keep_empty_features=True, random_state=seed)
    imp.fit(X_train)
    return imp.transform(X_target)


def ppm_impute(X_train, X_target, delta=0.0, n_imputations=5, max_iter=10, seed=42):
    """Pattern-mixture model (Little 1993) via MICE conditioned on missingness pattern.
    Each feature's imputation model includes the missingness indicators of all features,
    so the imputation distribution differs across patterns. A feature's own-missingness
    effect is not identifiable from data, so it is set by the sensitivity parameter delta
    (shift in training-SD units, delta-adjusted PPM; Ratitch et al. 2013). delta=0 -> MAR."""
    M_train = np.isnan(X_train).astype(float)
    M_target = np.isnan(X_target).astype(float)
    X_imp = mice_impute(X_train, X_target, n_imputations=n_imputations, max_iter=max_iter, seed=seed,
                        extra_train=M_train, extra_target=M_target)
    if delta != 0.0:
        sd = np.nan_to_num(np.nanstd(X_train, axis=0))
        X_imp = np.where(M_target.astype(bool), X_imp + delta * sd, X_imp)
    return X_imp


def baselines_impute(train_data, test_data=None, R_eval=None, n_neighbors=5, seed=42, ppm_delta=0.0):

    X_train = np.copy(train_data)
    is_evaluation_mode = (test_data is not None) and (R_eval is not None)

    if is_evaluation_mode:
        # evaluation mode: mask R_eval positions, then impute
        X_target = test_data.copy()
        X_target[R_eval] = np.nan
    else:
        # train mode: fill naturally missing values
        X_target = X_train.copy()

    results = {}

    mean_imp = SimpleImputer(strategy="mean")
    mean_imp.fit(X_train)
    results["Mean"] = mean_imp.transform(X_target)

    median_imp = SimpleImputer(strategy="median")
    median_imp.fit(X_train)
    results["Median"] = median_imp.transform(X_target)
    set_deterministic_seed(seed)
    knn1_imp = KNNImputer(n_neighbors=1)
    knn1_imp.fit(X_train)
    results["KNN1"] = knn1_imp.transform(X_target)

    set_deterministic_seed(seed)
    knn3_imp = KNNImputer(n_neighbors=3)
    knn3_imp.fit(X_train)
    results["KNN3"] = knn3_imp.transform(X_target)

    set_deterministic_seed(seed)
    knn_imp = KNNImputer(n_neighbors=n_neighbors)
    knn_imp.fit(X_train)
    results["KNN"] = knn_imp.transform(X_target)

    n_features = X_train.shape[1]
    X_lr = X_target.copy()
    col_indices = np.arange(n_features)

    for j in range(n_features):
        mask_missing = np.isnan(X_target[:, j])
        if not mask_missing.any(): continue

        other_cols = col_indices != j
        train_rows = ~np.isnan(X_train[:, j])
        y_train = X_train[train_rows, j]
        X_train_other = X_train[train_rows][:, other_cols]
        X_target_other = X_target[mask_missing][:, other_cols]
        train_means = np.nanmean(X_train_other, axis=0)
        X_train_other = np.nan_to_num(X_train_other, nan=train_means)
        X_target_other = np.nan_to_num(X_target_other, nan=train_means)
        set_deterministic_seed(seed)
        lr = LinearRegression()
        lr.fit(X_train_other, y_train)
        X_lr[mask_missing, j] = lr.predict(X_target_other)

    results["LinearRegression"] = X_lr

    set_deterministic_seed(seed)
    results["MICE"] = mice_impute(X_train, X_target, seed=seed)

    set_deterministic_seed(seed)
    results["MissForest"] = missforest_impute(X_train, X_target, seed=seed)

    set_deterministic_seed(seed)
    results["PPM"] = ppm_impute(X_train, X_target, delta=ppm_delta, seed=seed)

    return results


def main():
    parse_and_override(config.config)
    seed = config.config["random_state"]
    set_deterministic_seed(seed)
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    OUT = config.config["output_dir"]
    out_dir  = os.path.join(OUT, "output")
    imp_dir  = os.path.join(OUT, "imputations")
    emb_dir  = os.path.join(OUT, "embeddings")
    plot_dir = os.path.join(OUT, "plots")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(imp_dir,  exist_ok=True)
    os.makedirs(emb_dir,  exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    finetune_path = config.config.get("finetune_model_path", "")
    if finetune_path and os.path.exists(finetune_path):
        ckpt_path = finetune_path
        print(f"Loading finetuned checkpoint (Stage 2): {ckpt_path}")
    else:
        ckpt_path = config.config["save_model_path"]
        print(f"Loading Stage 1 checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fold_idx_for_eval = ckpt.get("fold_idx", 1)
    print(f"Loading checkpoint from fold {fold_idx_for_eval}")

    train_data, test_data, modality_dims = get_or_create_split(config.config, save=True)
    scalers, medians = load_global_scalers()

    # Load all modalities. Apply StandardScaler only to those in modalities_to_scale;
    # the rest (binary, categorical, pre-normalized like M5) are used as-is.
    modalities_to_scale = set(config.config['modalities_to_scale'])
    all_mods = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']
    train_modality_data = {}
    test_modality_data = {}
    for mod_key in all_mods:
        if mod_key in modalities_to_scale:
            train_modality_data[mod_key] = transform_modality(scalers, medians, train_data[mod_key], mod_key)
            test_modality_data[mod_key] = transform_modality(scalers, medians, test_data[mod_key], mod_key)
        else:
            train_modality_data[mod_key] = train_data[mod_key]
            test_modality_data[mod_key] = test_data[mod_key]

    train_ds = PatientDataset(train_modality_data, train_data["labels"], train_data["patient_ids"])
    train_loader = DataLoader(train_ds, batch_size=config.config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

    graph_builder = MultiModalGraphConstructor(
        modality_dims, hidden_dim=config.config["gcn_dim"], num_heads=config.config["graph_heads"],
        theta=config.config["theta"], top_k=config.config["top_k"], dropout=config.config["dropout"], aggregate=True,
        filter_graph=bool(config.config["filter_graph"]),
        reliable_ratio=bool(config.config["reliable_ratio"]),
    ).to(device)
    fusion = ModalityFusion(hidden_dim=config.config["gcn_dim"], mode=config.config["fusion_mode"]).to(device)
    gcn_encoder = PatientGCNEncoder(hidden_dim=config.config["gcn_dim"], num_layers=config.config["num_gcn_layers"], dropout=config.config["dropout"]).to(device)

    modalities_to_impute = config.config["modalities_to_impute"]
    print(f"\nModalities to impute: {modalities_to_impute}")
    decoders = {}
    for mod_key in modalities_to_impute:
        decoders[mod_key] = ModalityDecoder(config.config["gcn_dim"], modality_dims[mod_key]).to(device)

    def _load(module, state, name):
        inc = module.load_state_dict(state, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            print(f"[load_checkpoint] {name}: missing={inc.missing_keys}, unexpected={inc.unexpected_keys}")

    _load(graph_builder, ckpt["graph_builder"], "graph_builder")
    _load(fusion, ckpt["fusion"], "fusion")
    _load(gcn_encoder, ckpt["gcn_encoder"], "gcn_encoder")
    for m in decoders:
        _load(decoders[m], ckpt["decoders"][m], f"decoders.{m}")

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Fold: {fold_idx_for_eval}")
    print(f"  Loss: {ckpt.get('loss', float('nan')):.4f}")

    print("\nIMPUTING TRAINING SET (for classification)")
    print("=" * 60)

    print("\n[Model] Imputing training set")
    set_deterministic_seed(seed)
    train_results = model_impute(train_loader, graph_builder, fusion, gcn_encoder, decoders, None, device, apply_mask=False)

    np.save(os.path.join(emb_dir, "train_embeddings.npy"), train_results['embeddings'])
    np.save(os.path.join(emb_dir, "train_embedding_ids.npy"), train_results['embedding_patient_ids'])
    print(f"  Train embeddings: {train_results['embeddings'].shape} → {os.path.join(emb_dir, 'train_embeddings.npy')}")

    for mod_key in modalities_to_impute:
        pred_original = inverse_transform_modality(scalers, train_results[mod_key]['pred_std'], mod_key)
        if mod_key == 'm5': pred_original = round_m5_features(pred_original)
        # Use decoder output only for naturally missing features; keep original for observed.
        # Mirrors LR/KNN/GRAPE which only fill missing values — prevents data leakage.
        original = train_data[mod_key]   # NaN where missing, true values where observed
        combined = np.where(np.isnan(original), pred_original, original)
        np.save(os.path.join(imp_dir, f"train_model_{mod_key}.npy"), combined)
        print(f"  Saved: {os.path.join(imp_dir, f'train_model_{mod_key}.npy')}")

    print("\n[Baselines] Imputing training set")
    categorical_modalities = set(config.config['categorical_modalities'])

    for mod_key in modalities_to_impute:
        train_baselines_std = baselines_impute(train_modality_data[mod_key], n_neighbors=5, seed=seed, ppm_delta=config.config.get("ppm_delta", 0.0))

        for method, imputed_std in train_baselines_std.items():
            imputed = inverse_transform_modality(scalers, imputed_std, mod_key)
            if mod_key == 'm5': imputed = round_m5_features(imputed)
            if mod_key in categorical_modalities:
                imputed = to_onehot(imputed)
            np.save(os.path.join(imp_dir, f"train_{method.lower()}_{mod_key}.npy"), imputed)

    # Train GAIN for each continuous modality (m3, m5, m7).
    # since m6 is categorical (one-hot APOE), so GAIN is not applicable.
    print("\n[GAIN] Training models")
    gain_models = {}
    for mod_key in modalities_to_impute:
        if mod_key in categorical_modalities:
            continue
        print(f"  {mod_key.upper()}: fitting GAIN (10000 iterations)...")
        set_deterministic_seed(seed)
        model = GAINImputer(batch_size=128, hint_rate=0.9, alpha=100, iterations=10000, seed=seed)
        model.fit(train_modality_data[mod_key])
        gain_models[mod_key] = model

        gain_train_std = model.transform(train_modality_data[mod_key])
        gain_train = inverse_transform_modality(scalers, gain_train_std, mod_key)
        if mod_key == 'm5':
            gain_train = round_m5_features(gain_train)
        np.save(os.path.join(imp_dir, f"train_gain_{mod_key}.npy"), gain_train)
        print(f"  Saved: {os.path.join(imp_dir, f'train_gain_{mod_key}.npy')}")

    # m6 (categorical): copy mean imputation as a proxy so classification.py can use them
    for mod_key in categorical_modalities:
        if mod_key not in modalities_to_impute:
            continue
        src = os.path.join(imp_dir, f"train_mean_{mod_key}.npy")
        dst = os.path.join(imp_dir, f"train_gain_{mod_key}.npy")
        if os.path.exists(src):
            np.save(dst, np.load(src))
    print("  m6 (categorical): mean imputation used as proxy for GAIN")

    # Train GRAPE for each continuous modality (m3, m5, m7).
    # m6 (categorical one-hot APOE): GRAPE is not applicable; mean proxy used.
    print("\n[GRAPE] Training models")
    grape_models = {}
    for mod_key in modalities_to_impute:
        if mod_key in categorical_modalities:
            continue
        print(f"  {mod_key.upper()}: fitting GRAPE (3000 epochs)...")
        set_deterministic_seed(seed)
        model = GRAPEImputer(node_dim=64, edge_dim=64, n_layers=3, known=0.7,
                             epochs=3000, lr=1e-3, dropout=0.0, seed=seed)
        model.fit(train_modality_data[mod_key])
        grape_models[mod_key] = model

        grape_train_std = model.transform(train_modality_data[mod_key])
        grape_train = inverse_transform_modality(scalers, grape_train_std, mod_key)
        if mod_key == 'm5':
            grape_train = round_m5_features(grape_train)
        np.save(os.path.join(imp_dir, f"train_grape_{mod_key}.npy"), grape_train)
        print(f"  Saved: {os.path.join(imp_dir, f'train_grape_{mod_key}.npy')}")

    for mod_key in categorical_modalities:
        if mod_key not in modalities_to_impute:
            continue
        src = os.path.join(imp_dir, f"train_mean_{mod_key}.npy")
        dst = os.path.join(imp_dir, f"train_grape_{mod_key}.npy")
        if os.path.exists(src):
            np.save(dst, np.load(src))
    print("  m6 (categorical): mean imputation used as proxy for GRAPE")

    print("\nGENERATING TEST EMBEDDINGS")
    print("=" * 60)
    set_deterministic_seed(seed)
    test_embed_results = model_impute_single(train_modality_data, test_modality_data, test_data["patient_ids"], graph_builder, fusion, gcn_encoder, decoders, None, device, apply_mask=False)

    np.save(os.path.join(emb_dir, "test_embeddings.npy"), test_embed_results['embeddings'])
    np.save(os.path.join(emb_dir, "test_embedding_ids.npy"), test_embed_results['embedding_patient_ids'])
    print(f"  Test embeddings: {test_embed_results['embeddings'].shape} → {os.path.join(emb_dir, 'test_embeddings.npy')}")

    print("\nIMPUTING TEST SET (for imputation evaluation)")
    print("=" * 60)

    mask_probs = {
        'm3': config.config["eval_prob_m3"], 'm5': config.config["eval_prob_m5"],
        'm6': config.config["eval_prob_m6"], 'm7': config.config["eval_prob_m7"],
    }

    R_eval_dict = {}
    for mod_key in modalities_to_impute:
        mask_prob = mask_probs[mod_key]
        R_eval_dict[mod_key] = get_or_create_eval_mask(test_data[mod_key], modality_key=mod_key, mask_prob=mask_prob, seed=seed)
        print(f"R_eval mask for {mod_key.upper()}: prob={mask_prob}")

    print("\n[Model] Forward pass with R_eval masking")
    set_deterministic_seed(seed)
    test_results = model_impute_single(train_modality_data, test_modality_data, test_data["patient_ids"], graph_builder, fusion, gcn_encoder, decoders, R_eval_dict, device, apply_mask=True)

    print("\n[Baselines] Imputing test set")
    baseline_results = {}

    for mod_key in modalities_to_impute:
        baseline_results[mod_key] = baselines_impute(train_modality_data[mod_key], test_modality_data[mod_key], R_eval_dict[mod_key], n_neighbors=5, seed=seed, ppm_delta=config.config.get("ppm_delta", 0.0))

    # GAIN test imputation: mask R_eval positions, then impute using the already-trained model.
    print("[GAIN] Imputing test set")
    for mod_key in modalities_to_impute:
        if mod_key not in gain_models:
            continue
        X_target = test_modality_data[mod_key].copy()
        X_target[R_eval_dict[mod_key]] = np.nan
        gain_test_std = gain_models[mod_key].transform(X_target)
        baseline_results[mod_key]["GAIN"] = gain_test_std

    # GRAPE test imputation: mask R_eval positions, then impute using the already-trained model.
    print("[GRAPE] Imputing test set")
    for mod_key in modalities_to_impute:
        if mod_key not in grape_models:
            continue
        X_target = test_modality_data[mod_key].copy()
        X_target[R_eval_dict[mod_key]] = np.nan
        grape_test_std = grape_models[mod_key].transform(X_target)
        baseline_results[mod_key]["GRAPE"] = grape_test_std

    for mod_key in modalities_to_impute:
        pred_std_raw = test_results[mod_key]['pred_std']
        np.save(os.path.join(imp_dir, f"test_model_{mod_key}_raw.npy"), pred_std_raw)
        pred_original = inverse_transform_modality(scalers, pred_std_raw, mod_key)  # original scale
        if mod_key == 'm5': pred_original = round_m5_features(pred_original)
        np.save(os.path.join(imp_dir, f"test_model_{mod_key}.npy"), pred_original)

        for method, imputed_std in baseline_results[mod_key].items():
            pred_original = inverse_transform_modality(scalers, imputed_std, mod_key)
            if mod_key == 'm5': pred_original = round_m5_features(pred_original)
            if mod_key in categorical_modalities:
                pred_original = to_onehot(pred_original)
            np.save(os.path.join(imp_dir, f"test_{method.lower()}_{mod_key}.npy"), pred_original)

    # m6 test proxy: copy mean imputation for GAIN and GRAPE.
    for mod_key in categorical_modalities:
        if mod_key not in modalities_to_impute:
            continue
        src = os.path.join(imp_dir, f"test_mean_{mod_key}.npy")
        for dst in [os.path.join(imp_dir, f"test_gain_{mod_key}.npy"),
                    os.path.join(imp_dir, f"test_grape_{mod_key}.npy")]:
            if os.path.exists(src):
                np.save(dst, np.load(src))

    print("\nImputation Quality Metrics (on hidden test positions)")
    print("=" * 80)
    feature_names = {
        'm3': ['TotalChol', 'HDL', 'Glucose', 'HbA1c', 'BMI', 'SBP', 'DBP'],
        'm5': ["MMSE", "MOCATOTS", "CDRSUM", "CDRGLOB", "MINTTOTS", "CRAFTURS", "REY1REC", "GDS", "MOCA_CLOCK_TOTAL"],
        'm6': ["APOE_22", "APOE_23", "APOE_24", "APOE_33", "APOE_34", "APOE_44"],
        'm7': ["Ttau", "AB42", "AB40", "AB4240ratio", "Ptau181", "NFL", "GFAP", "Ptau217"]
    }

    all_results = {}
    for mod_key in modalities_to_impute:
        print(f"\nEVALUATING {mod_key.upper()}")
        print(f"{'=' * 80}")

        true_std = test_results[mod_key]['true_std']
        pred_std = test_results[mod_key]['pred_std']
        patient_ids = test_results[mod_key]['patient_ids']
        R_eval = R_eval_dict[mod_key]

        original_key = f'{mod_key}_original'
        if original_key in test_data:
            true_original = test_data[original_key].copy()
            print(f"  Using unclipped original values as ground truth for {mod_key.upper()}")
        else:
            true_original = inverse_transform_modality(scalers, true_std, mod_key)
        pred_original = inverse_transform_modality(scalers, pred_std, mod_key)
        if mod_key == 'm5': pred_original = round_m5_features(pred_original)

        baseline_original = {}
        baseline_scores = {}  # raw scores for AUROC
        for method in baseline_results[mod_key].keys():
            bl_pred = inverse_transform_modality(scalers, baseline_results[mod_key][method], mod_key)
            if mod_key == 'm5': bl_pred = round_m5_features(bl_pred)
            if mod_key in categorical_modalities:
                baseline_scores[method] = bl_pred
                bl_pred = to_onehot(bl_pred)
            baseline_original[method] = bl_pred

        feat_names = feature_names.get(mod_key, [f"Feature_{i}" for i in range(true_original.shape[1])])
        feat_names = feat_names[:true_original.shape[1]]
        methods_to_eval = ["Model"] + sorted(baseline_original.keys())

        if mod_key in categorical_modalities:
            model_scores = test_results[mod_key].get('pred_scores', None)
            cat_results = evaluate_categorical(mod_key, true_std, true_original, pred_original, baseline_original, R_eval, feat_names, methods_to_eval, model_scores=model_scores, baseline_scores=baseline_scores)
            if cat_results is None:
                continue

            all_results[mod_key] = {'type': 'categorical', 'feat_names': feat_names, 'methods': methods_to_eval, 'cat_results': cat_results,}
            plot_categorical_metrics(mod_key, methods_to_eval, cat_results, save_dir=plot_dir)

        else:
            perfeat_results, patient_mae, feat_n = evaluate_continuous(mod_key, true_original, pred_original, baseline_original, R_eval, feat_names, methods_to_eval, patient_ids)

            all_results[mod_key] = {'type': 'continuous', 'feat_names': feat_names, 'methods': methods_to_eval, 'perfeat_results': perfeat_results, 'patient_mae': patient_mae}

            for metric in ["mae", "mse", "rmse", "mape"]:
                data = {method: perfeat_results[method][metric] for method in methods_to_eval}
                plot_bars_by_feature(data, feat_names, os.path.join(plot_dir, f"{metric}_by_feature_{mod_key}.png"), f"{metric.upper()} by Feature - {mod_key.upper()}")

            # per-feature plots
            os.makedirs(os.path.join(plot_dir, mod_key), exist_ok=True)
            for j, feat_name in enumerate(feat_names):
                for metric in ["mae", "mse", "rmse", "mape"]:
                    values = [perfeat_results[method][metric][j] for method in methods_to_eval]
                    if all(np.isnan(v) for v in values):
                        continue  # skip if all NaN
                    outfile = os.path.join(plot_dir, mod_key, f"{feat_name}_{metric}.png")
                    title = f"{feat_name} - {metric.upper()} ({mod_key.upper()})"
                    plot_single_feature(methods_to_eval, values, outfile, title, metric.upper(), n=feat_n[j])

            # Trimmed evaluation for modalities with outlier clipping (e.g. M7)
            outlier_clip_mods = set(config.config.get('outlier_clip', {}).keys())
            if mod_key in outlier_clip_mods:
                trim_pct = 95
                print(f"\n{'=' * 80}")
                print(f"TRIMMED EVALUATION {mod_key.upper()} (excluding top {100 - trim_pct}% outliers per feature)")
                print(f"{'=' * 80}")
                trimmed_results, trimmed_patient_mae, trimmed_n = evaluate_continuous_trimmed(
                    mod_key, true_original, pred_original, baseline_original, R_eval,
                    feat_names, methods_to_eval, patient_ids, trim_percentile=trim_pct)

                all_results[f'{mod_key}_trimmed'] = {
                    'type': 'continuous', 'feat_names': feat_names, 'methods': methods_to_eval,
                    'perfeat_results': trimmed_results, 'patient_mae': trimmed_patient_mae}

                for metric in ["mae", "mse", "rmse", "mape"]:
                    data = {method: trimmed_results[method][metric] for method in methods_to_eval}
                    plot_bars_by_feature(data, feat_names,
                        os.path.join(plot_dir, f"{metric}_by_feature_{mod_key}_trimmed.png"),
                        f"{metric.upper()} by Feature - {mod_key.upper()} (Trimmed p{trim_pct})")

                trimmed_dir = os.path.join(plot_dir, f"{mod_key}_trimmed")
                os.makedirs(trimmed_dir, exist_ok=True)
                for j, feat_name in enumerate(feat_names):
                    for metric in ["mae", "mse", "rmse", "mape"]:
                        values = [trimmed_results[method][metric][j] for method in methods_to_eval]
                        if all(np.isnan(v) for v in values):
                            continue
                        outfile = os.path.join(trimmed_dir, f"{feat_name}_{metric}.png")
                        title = f"{feat_name} - {metric.upper()} ({mod_key.upper()}, Trimmed p{trim_pct})"
                        plot_single_feature(methods_to_eval, values, outfile, title, metric.upper(), n=trimmed_n[j])

    save_results_csv(all_results, os.path.join(out_dir, "imputation_all_modalities.csv"))


if __name__ == "__main__":
    main()
