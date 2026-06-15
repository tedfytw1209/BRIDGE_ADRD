import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import argparse

import config
from data_augmentation import compute_imputation_loss
from data_loader import PatientDataset, get_or_create_split, load_global_scalers
from decoders import ModalityDecoder
from fusion import ModalityFusion
from gcn import PatientGCNEncoder
from graph_builder import MultiModalGraphConstructor
from utils import set_deterministic_seed, transform_modality


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


def load_stage1_checkpoint(ckpt_path, modality_dims, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    graph_builder = MultiModalGraphConstructor(
        modality_dims, hidden_dim=config.config["gcn_dim"], num_heads=config.config["graph_heads"],
        theta=config.config["theta"], top_k=config.config["top_k"], dropout=config.config["dropout"], aggregate=True,
        filter_graph=bool(config.config["filter_graph"]),
        reliable_ratio=bool(config.config["reliable_ratio"]),
    ).to(device)

    fusion = ModalityFusion(hidden_dim=config.config["gcn_dim"], mode=config.config["fusion_mode"]).to(device)
    gcn_encoder = PatientGCNEncoder(hidden_dim=config.config["gcn_dim"], num_layers=config.config["num_gcn_layers"], dropout=config.config["dropout"],).to(device)
    decoders = {m: ModalityDecoder(config.config["gcn_dim"], modality_dims[m]).to(device) for m in config.config["modalities_to_impute"]}

    def _load(module, state, name):
        inc = module.load_state_dict(state, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            print(f"  {name}: missing={inc.missing_keys}, unexpected={inc.unexpected_keys}")

    _load(graph_builder, ckpt["graph_builder"], "graph_builder")
    _load(fusion, ckpt["fusion"], "fusion")
    _load(gcn_encoder, ckpt["gcn_encoder"], "gcn_encoder")
    for m in decoders:
        _load(decoders[m], ckpt["decoders"][m], f"decoders.{m}")

    print(f"Loaded Stage 1 checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch','?')}, fold {ckpt.get('fold_idx','?')})")
    return graph_builder, fusion, gcn_encoder, decoders, ckpt


def _make_feature_masks(x_dict_filled, feat_mask_dict, categorical_modalities, device):
    x_tilde_dict = {}
    feat_mask_tilde_dict = {}
    R_dict = {}

    for m in sorted(x_dict_filled.keys()):
        x_m = x_dict_filled[m]
        feat_mask = feat_mask_dict[m]
        prob = config.config[f"prob_{m}"]

        if m in categorical_modalities:
            patient_keep = torch.bernoulli(torch.ones(x_m.size(0), 1, device=device) * (1 - prob))
            keep = patient_keep.expand_as(x_m)
        else:
            keep = torch.bernoulli(torch.ones_like(x_m) * (1 - prob))

        R_dict[m] = (1 - keep) * feat_mask  # observed features that were randomly hidden
        x_tilde_dict[m] = x_m * keep
        feat_mask_tilde_dict[m] = feat_mask * keep

    return x_tilde_dict, feat_mask_tilde_dict, R_dict


def finetune_epoch(graph_builder, fusion, gcn_encoder, decoders, loader, optimizer, device, categorical_modalities):
    graph_builder.eval(); fusion.eval(); gcn_encoder.eval()
    for d in decoders.values():
        d.train()

    total_loss = 0.0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        available_modalities = sorted([f"m{i}" for i in range(1, 8) if f"m{i}" in batch])
        x_dict = {m: batch[m] for m in available_modalities}
        feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in x_dict}
        x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in x_dict}

        x_tilde, feat_mask_tilde, R_dict = _make_feature_masks(x_dict_filled, feat_mask_dict, categorical_modalities, device)
        patient_mask_tilde = {m: (feat_mask_tilde[m].sum(dim=1) > 0).float() for m in available_modalities}

        with torch.no_grad():
            A_hat, modality_info = graph_builder(x_tilde, feat_mask_tilde, patient_mask_tilde)
            h_m_dict = {m: modality_info[f"h_m_{m}"] for m in available_modalities}
            if config.config["skip_graph"]:
                h_hat_m_dict = h_m_dict
            else:
                _pmask = patient_mask_tilde if config.config["modality_a"] else None
                h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)
            h_fused_dict = fusion(h_m_dict, h_hat_m_dict, patient_mask_tilde)

        x_hat_dict = {}
        R_imp_dict = {}
        for m in decoders:
            if m in h_fused_dict and R_dict[m].sum() > 0:
                x_hat_dict[m] = decoders[m](h_fused_dict[m], x_tilde[m], feat_mask_tilde[m])
                R_imp_dict[m] = R_dict[m]

        if not R_imp_dict:
            continue

        L_imp = compute_imputation_loss(x_dict_filled, x_hat_dict, R_imp_dict, categorical_modalities=list(categorical_modalities))

        optimizer.zero_grad()
        L_imp.backward()
        torch.nn.utils.clip_grad_norm_([p for d in decoders.values() for p in d.parameters()], max_norm=5.0)
        optimizer.step()
        total_loss += L_imp.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(graph_builder, fusion, gcn_encoder, decoders, loader, device, categorical_modalities):
    graph_builder.eval(); fusion.eval(); gcn_encoder.eval()
    for d in decoders.values():
        d.eval()

    val_seed = config.config["random_state"]
    set_deterministic_seed(val_seed)

    total_loss = 0.0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        available_modalities = sorted([f"m{i}" for i in range(1, 8) if f"m{i}" in batch])
        x_dict = {m: batch[m] for m in available_modalities}
        feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in x_dict}
        x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in x_dict}

        x_tilde, feat_mask_tilde, R_dict = _make_feature_masks(x_dict_filled, feat_mask_dict, categorical_modalities, device)
        patient_mask_tilde = {m: (feat_mask_tilde[m].sum(dim=1) > 0).float() for m in available_modalities}

        A_hat, modality_info = graph_builder(x_tilde, feat_mask_tilde, patient_mask_tilde)
        h_m_dict = {m: modality_info[f"h_m_{m}"] for m in available_modalities}
        if config.config["skip_graph"]:
            h_hat_m_dict = h_m_dict
        else:
            _pmask = patient_mask_tilde if config.config["modality_a"] else None
            h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)
        h_fused_dict = fusion(h_m_dict, h_hat_m_dict, patient_mask_tilde)

        x_hat_dict = {}
        R_imp_dict = {}
        for m in decoders:
            if m in h_fused_dict and R_dict[m].sum() > 0:
                x_hat_dict[m] = decoders[m](h_fused_dict[m], x_tilde[m], feat_mask_tilde[m])
                R_imp_dict[m] = R_dict[m]

        if not R_imp_dict:
            continue

        L_imp = compute_imputation_loss(x_dict_filled, x_hat_dict, R_imp_dict, categorical_modalities=list(categorical_modalities))
        total_loss += L_imp.item()

    return total_loss / max(len(loader), 1)


def main():
    parse_and_override(config.config)
    seed = config.config["random_state"]
    set_deterministic_seed(seed)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    os.makedirs("output", exist_ok=True)

    stage1_path   = config.config["save_model_path"]
    finetune_path = config.config["finetune_model_path"]

    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_path}. Run train.py first.")

    backup_path = stage1_path.replace(".pth", "_stage1.pth")
    shutil.copy2(stage1_path, backup_path)
    print(f"Stage 1 checkpoint backed up to: {backup_path}")

    train_data, _, modality_dims = get_or_create_split(config.config, save=True)
    scalers, medians = load_global_scalers()

    modalities_to_scale = set(config.config["modalities_to_scale"])
    train_modality_data = {}
    for m in [f"m{i}" for i in range(1, 8)]:
        if m in modalities_to_scale:
            train_modality_data[m] = transform_modality(scalers, medians, train_data[m], m)
        else:
            train_modality_data[m] = train_data[m]

    labels = np.asarray(train_data["labels"])
    all_idx = np.arange(len(labels))
    val_frac = config.config["finetune_val_frac"]
    train_idx, val_idx = train_test_split(all_idx, test_size=val_frac, stratify=labels, random_state=seed)

    def _make_loader(idx, shuffle):
        subset = {m: train_modality_data[m][idx] for m in train_modality_data}
        lbl = labels[idx]
        pids = np.asarray(train_data["patient_ids"])[idx]
        ds = PatientDataset(subset, lbl, pids)
        return DataLoader(ds, batch_size=config.config["batch_size"], shuffle=shuffle, num_workers=0, pin_memory=False)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader   = _make_loader(val_idx,   shuffle=False)
    print(f"Finetune split: train={len(train_idx)}, val={len(val_idx)}")

    graph_builder, fusion, gcn_encoder, decoders, stage1_ckpt = load_stage1_checkpoint(stage1_path, modality_dims, device)

    # Freeze GCN - only decoder params receive gradients
    for param in (list(graph_builder.parameters()) + list(fusion.parameters()) + list(gcn_encoder.parameters())):
        param.requires_grad = False

    optimizer = optim.Adam([p for d in decoders.values() for p in d.parameters()], lr=config.config["finetune_lr"])

    categorical_modalities = set(config.config["categorical_modalities"])
    best_val = float("inf")
    best_epoch = 0
    bad = 0
    patience = config.config["finetune_patience"]
    print(f"\nFinetuning decoder for up to {config.config['finetune_epochs']} epochs (patience={patience}, lr={config.config['finetune_lr']})")

    best_decoder_states = {m: {k: v.clone() for k, v in decoders[m].state_dict().items()} for m in decoders}
    for epoch in range(1, config.config["finetune_epochs"] + 1):
        set_deterministic_seed(seed + epoch) #more deterministic across epochs
        train_loss = finetune_epoch(graph_builder, fusion, gcn_encoder, decoders, train_loader, optimizer, device, categorical_modalities)
        val_loss = val_epoch(graph_builder, fusion, gcn_encoder, decoders, val_loader, device, categorical_modalities)
        print(f"Epoch {epoch:03d} | train_L_imp={train_loss:.4f}  val_L_imp={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            bad = 0
            best_decoder_states = {m: {k: v.clone() for k, v in decoders[m].state_dict().items()} for m in decoders}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val={best_val:.4f})")
                break

    for m in decoders:
        decoders[m].load_state_dict(best_decoder_states[m])

    finetuned_ckpt = {k: v for k, v in stage1_ckpt.items()}
    finetuned_ckpt["decoders"] = {m: decoders[m].state_dict() for m in decoders}
    finetuned_ckpt["finetune_epoch"] = best_epoch
    finetuned_ckpt["finetune_val_loss"] = best_val

    torch.save(finetuned_ckpt, finetune_path)
    print(f"\nFinetuned checkpoint saved to: {finetune_path}")
    print(f"  Best finetune epoch: {best_epoch}, val L_imp: {best_val:.4f}")


if __name__ == "__main__":
    main()
