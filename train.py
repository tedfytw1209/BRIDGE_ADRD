import os
import sys
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_loader import get_or_create_split, create_fold_dataloaders, fit_global_scalers, apply_global_scalers
from graph_builder import MultiModalGraphConstructor
from gcn import PatientGCNEncoder
from data_augmentation import multimodal_data_augmentation, contrastive_loss, compute_imputation_loss, compute_joint_loss
from decoders import ModalityDecoder
from fusion import ModalityFusion
from utils import set_deterministic_seed
import config
import argparse


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


def tprint(*args, **kwargs):
    print(*args, **kwargs)
    kwargs.pop("file", None)
    print(*args, **kwargs, file=sys.__stdout__)
    sys.__stdout__.flush()


def train_one_epoch(graph_builder, fusion, gcn_encoder, decoders, train_loader, optimizer, device, lambda_imp):

    graph_builder.train(); fusion.train(); gcn_encoder.train()
    for d in decoders.values(): d.train()

    total_loss = 0.0

    for batch in train_loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        available_modalities = sorted([f'm{i}' for i in range(1, 8) if f'm{i}' in batch])
        x_dict = {m: batch[m] for m in available_modalities}
        feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in x_dict}
        x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in x_dict}

        # Patient-level masks (1 if patient has any features in this modality)
        patient_mask_dict = {m: (feat_mask_dict[m].sum(dim=1) > 0).float() for m in x_dict}

        # ===== VIEW A: ORIGINAL (G) =====
        # Build graph once and get A_hat (combined adjacency) and h^m (encoder representations)
        # h^m = f^m_Φ(x^m), A^m via attention, A_hat = aggregated
        A_hat, modality_info = graph_builder(x_dict_filled, feat_mask_dict=feat_mask_dict, patient_mask_dict=patient_mask_dict)
        h_m_dict = {m: modality_info[f"h_m_{m}"] for m in available_modalities}

        # h_hat^m = GCN(h^m, A_hat) - message passing on original graph
        if config.config["skip_graph"]:
            h_hat_m_dict = h_m_dict
        else:
            _pmask = patient_mask_dict if config.config["modality_a"] else None
            h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)

        # Fusion: h_fused = α·h^m + β·h_hat^m (or h_hat^m if missing)
        h_fused_dict = fusion(h_m_dict, h_hat_m_dict, patient_mask_dict)

        # z = concatenation of all fused representations (patient embedding)
        z = torch.cat([h_fused_dict[m] for m in available_modalities], dim=1)

        # ===== DATA AUGMENTATION =====
        # Feature masking: x_tilde = x * feature_mask
        # Edge dropping: A_tilde = A_hat * edge_mask
        x_tilde_dict, feat_mask_tilde_dict, A_tilde, R_dict = multimodal_data_augmentation(x_dict_filled, feat_mask_dict, A_hat)

        # Patient-level masks for augmented view
        patient_mask_tilde_dict = {m: (feat_mask_tilde_dict[m].sum(dim=1) > 0).float() for m in available_modalities}

        # ===== VIEW B: AUGMENTED (G_tilde) =====
        # Encode augmented features using the SAME encoders
        h_m_tilde_dict = {}
        for m in available_modalities:
            h_m_tilde_dict[m] = graph_builder.encoders[m](x_tilde_dict[m], feat_mask_tilde_dict[m])

        # GCN with edge-dropped ORIGINAL adjacency (A_tilde)
        if config.config["skip_graph"]:
            h_hat_m_tilde_dict = h_m_tilde_dict
        else:
            _pmask_t = patient_mask_tilde_dict if config.config["modality_a"] else None
            h_hat_m_tilde_dict = gcn_encoder(h_m_tilde_dict, A_tilde, _pmask_t)

        # Fusion for augmented view
        h_fused_tilde_dict = fusion(h_m_tilde_dict, h_hat_m_tilde_dict, patient_mask_tilde_dict)

        # z_tilde = augmented view patient embedding
        z_tilde = torch.cat([h_fused_tilde_dict[m] for m in available_modalities], dim=1)

        # ===== IMPUTATION LOSS (L^imp) =====
        # Decode from AUGMENTED view's fused representations (h_fused_tilde)
        # It forces the decoder to learn from NEIGHBORS since R_dict values are masked in augmented view
        # x_hat^m = decoder(h_fused_tilde^m)
        x_hat_dict = {}
        R_imp_dict = {}
        for m in decoders.keys():
            # Pass augmented peer feature values and mask so decoder sees observed
            # peers directly (like LR/KNN) while keeping GCN cross-patient context.
            x_hat_dict[m] = decoders[m](h_fused_tilde_dict[m], x_tilde_dict[m], feat_mask_tilde_dict[m])
            R_imp_dict[m] = R_dict[m]

        # Compare predicted (from augmented) with original values at masked positions
        L_imp = compute_imputation_loss(x_dict_filled, x_hat_dict, R_imp_dict, categorical_modalities=config.config['categorical_modalities'])

        # ===== CONTRASTIVE LOSS (L^gcl) =====
        # Bidirectional: L = α1·L_fwd(z, z_tilde, A_hat) + α2·L_bwd(z_tilde, z, A_tilde)
        L_gcl = contrastive_loss(z, z_tilde, A_hat, A_tilde)

        # ===== JOINT LOSS: L = L^gcl + λ·L^imp =====
        L = compute_joint_loss(L_gcl, L_imp, lambda_imp, lambda_cl=config.config["lambda_cl"])

        # BACKWARD & OPTIMIZE
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(
            list(graph_builder.parameters()) + list(fusion.parameters()) + list(gcn_encoder.parameters()) +
            [p for d in decoders.values() for p in d.parameters()], max_norm=5.0
        )
        optimizer.step()

        total_loss += L.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_val_epoch(graph_builder, fusion, gcn_encoder, decoders, val_loader, device, lambda_imp):
    """Validation epoch with deterministic augmentation."""

    graph_builder.eval(); fusion.eval(); gcn_encoder.eval()
    for d in decoders.values(): d.eval()

    total_loss, total_L_gcl, total_L_imp = 0.0, 0.0, 0.0

    val_seed = config.config["random_state"]
    set_deterministic_seed(val_seed)

    for batch in val_loader:

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        available_modalities = sorted([f'm{i}' for i in range(1, 8) if f'm{i}' in batch])
        x_dict = {m: batch[m] for m in available_modalities}
        feat_mask_dict = {m: torch.isfinite(x_dict[m]).float() for m in x_dict}
        x_dict_filled = {m: torch.nan_to_num(x_dict[m], nan=0.0) for m in x_dict}

        # Patient-level masks
        patient_mask_dict = {m: (feat_mask_dict[m].sum(dim=1) > 0).float() for m in x_dict}

        # ===== VIEW A: ORIGINAL (G) =====
        A_hat, modality_info = graph_builder(x_dict_filled, feat_mask_dict, patient_mask_dict)
        h_m_dict = {m: modality_info[f"h_m_{m}"] for m in available_modalities}
        if config.config["skip_graph"]:
            h_hat_m_dict = h_m_dict
        else:
            _pmask = patient_mask_dict if config.config["modality_a"] else None
            h_hat_m_dict = gcn_encoder(h_m_dict, A_hat, _pmask)
        h_fused_dict = fusion(h_m_dict, h_hat_m_dict, patient_mask_dict)

        z = torch.cat([h_fused_dict[m] for m in available_modalities], dim=1)

        # ===== DATA AUGMENTATION =====
        x_tilde_dict, feat_mask_tilde_dict, A_tilde, R_dict = multimodal_data_augmentation(x_dict_filled, feat_mask_dict, A_hat)

        patient_mask_tilde_dict = {m: (feat_mask_tilde_dict[m].sum(dim=1) > 0).float() for m in available_modalities}

        # ===== VIEW B: AUGMENTED (G_tilde) =====
        # Only encode augmented features, do NOT rebuild graph
        h_m_tilde_dict = {}
        for m in available_modalities:
            h_m_tilde_dict[m] = graph_builder.encoders[m](x_tilde_dict[m], feat_mask_tilde_dict[m])

        # GCN with edge-dropped ORIGINAL adjacency
        if config.config["skip_graph"]:
            h_hat_m_tilde_dict = h_m_tilde_dict
        else:
            _pmask_t = patient_mask_tilde_dict if config.config["modality_a"] else None
            h_hat_m_tilde_dict = gcn_encoder(h_m_tilde_dict, A_tilde, _pmask_t)
        h_fused_tilde_dict = fusion(h_m_tilde_dict, h_hat_m_tilde_dict, patient_mask_tilde_dict)

        z_tilde = torch.cat([h_fused_tilde_dict[m] for m in available_modalities], dim=1)

        # ===== IMPUTATION LOSS (L^imp) =====
        # Decode from AUGMENTED view (same as training)
        x_hat_dict = {}
        R_imp_dict = {}
        for m in decoders.keys():
            x_hat_dict[m] = decoders[m](h_fused_tilde_dict[m], x_tilde_dict[m], feat_mask_tilde_dict[m])
            R_imp_dict[m] = R_dict[m]

        L_imp = compute_imputation_loss(x_dict_filled, x_hat_dict, R_imp_dict, categorical_modalities=config.config['categorical_modalities'])

        # ===== CONTRASTIVE LOSS (L^gcl) =====
        L_gcl = contrastive_loss(z, z_tilde, A_hat, A_tilde)

        # ===== JOINT LOSS =====
        L = compute_joint_loss(L_gcl, L_imp, lambda_imp, lambda_cl=config.config["lambda_cl"])

        total_loss += float(L)
        total_L_gcl += float(L_gcl)
        total_L_imp += float(L_imp)

    n_batches = len(val_loader)
    return {
        "val_total": total_loss / n_batches,
        "val_contrastive": total_L_gcl / n_batches,
        "val_imputation": total_L_imp / n_batches,
    }


def save_checkpoint(epoch, graph_builder, fusion, gcn_encoder, decoders, optimizer, loss, path, fold_idx=None):
    state = {
        "epoch": epoch,
        "fold_idx": fold_idx,
        "graph_builder": graph_builder.state_dict(),
        "fusion": fusion.state_dict(),
        "gcn_encoder": gcn_encoder.state_dict(),
        "decoders": {m: d.state_dict() for m, d in decoders.items()},
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, path)
    print(f"Model saved at {path}")


def main():
    parse_and_override(config.config)
    out_dir = os.path.join(config.config["output_dir"], "output")
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = open(os.path.join(out_dir, "output.txt"), "w")
    seed = config.config["random_state"]
    set_deterministic_seed(seed)
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tprint(f"Using device: {device}")

    # ----- Data split: 80% train / 20% test -----
    train_data, test_data, modality_dims = get_or_create_split(config.config, save=True)
    modalities_to_scale = config.config["modalities_to_scale"]
    scalers, medians = fit_global_scalers(train_data, modalities_to_scale=modalities_to_scale)
    scaled_train = apply_global_scalers(train_data, scalers, medians, modalities_to_scale=modalities_to_scale)
    scaled_test = apply_global_scalers(test_data, scalers, medians, modalities_to_scale=modalities_to_scale)
    train_data = {**train_data, **scaled_train}
    test_data = {**test_data, **scaled_test}
    tprint(f"Total patients: {len(train_data['labels']) + len(test_data['labels'])}")
    tprint(f"Outer split → train N={len(train_data['labels'])}, test N={len(test_data['labels'])}")

    labels = np.asarray(train_data["labels"])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.config["random_state"])

    fold_best_vals = []
    global_best_val = float("inf")
    final_path = config.config["save_model_path"]

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels), start=1):
        tprint(f"\n==== Fold {fold_id}/10 ====")
        print(f"Train N={len(train_idx)}, Val N={len(val_idx)}\n")

        train_loader, val_loader = create_fold_dataloaders(
            modality_data=train_data, labels=train_data["labels"],
            ids=train_data["patient_ids"], train_idx=train_idx, val_idx=val_idx,
            batch_size=config.config["batch_size"]
        )

        graph_builder = MultiModalGraphConstructor(
            modality_dims, hidden_dim=config.config["gcn_dim"], num_heads=config.config["graph_heads"],
            theta=config.config["theta"], top_k=config.config["top_k"], dropout=config.config["dropout"], aggregate=True,
            filter_graph=bool(config.config["filter_graph"]),
            reliable_ratio=bool(config.config["reliable_ratio"]),
        ).to(device)
        gcn_encoder = PatientGCNEncoder(hidden_dim=config.config["gcn_dim"], num_layers=config.config["num_gcn_layers"], dropout=config.config["dropout"]).to(device)
        fusion = ModalityFusion(hidden_dim=config.config["gcn_dim"], mode=config.config["fusion_mode"]).to(device)

        # Decoders only for modalities that need imputation
        modalities_to_impute = config.config["modalities_to_impute"]
        decoders = {}
        for mod_key in modalities_to_impute:
            decoders[mod_key] = ModalityDecoder(config.config["gcn_dim"], modality_dims[mod_key]).to(device)

        params = (list(graph_builder.parameters()) + list(fusion.parameters()) + list(gcn_encoder.parameters()) + [p for d in decoders.values() for p in d.parameters()])
        optimizer = optim.Adam(params, lr=config.config["lr"], weight_decay=config.config["weight_decay"])

        # ----- Train epochs with val each epoch; save best on val_total -----
        best_val = float("inf")
        best_val_c = 0.0
        best_val_i = 0.0
        best_epoch = 0
        bad = 0
        patience = config.config["patience"]

        for epoch in range(1, config.config["epochs"] + 1):
            train_loss = train_one_epoch(graph_builder, fusion, gcn_encoder, decoders, train_loader, optimizer, device, config.config["lambda_imp"])

            # Validation
            val_metrics = evaluate_val_epoch(graph_builder, fusion, gcn_encoder, decoders, val_loader, device, config.config["lambda_imp"])
            print(f"Fold {fold_id} | Epoch {epoch:02d} | train={train_loss:.4f}, val={val_metrics['val_total']:.4f} (c={val_metrics['val_contrastive']:.4f}, i={val_metrics['val_imputation']:.4f})")

            # Early stop on best val_total
            if val_metrics["val_total"] < best_val - 1e-6:
                best_val = val_metrics["val_total"]
                best_val_c = val_metrics["val_contrastive"]
                best_val_i = val_metrics["val_imputation"]
                best_epoch = epoch
                bad = 0
                # Save to disk only when this is also the global best across all folds
                if best_val < global_best_val:
                    global_best_val = best_val
                    save_checkpoint(epoch, graph_builder, fusion, gcn_encoder, decoders, optimizer, best_val, path=final_path, fold_idx=fold_id)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Fold {fold_id}: early stopping at epoch {epoch} (best epoch {best_epoch}).")
                    break

        fold_best_vals.append(best_val)
        tprint(f"Fold {fold_id} done - epoch {best_epoch}, val={best_val:.4f} (c={best_val_c:.4f}, i={best_val_i:.4f})")

    # ----- CV SUMMARY -----
    mean_val = float(np.mean(fold_best_vals))
    std_val = float(np.std(fold_best_vals))
    best_fold = int(np.argmin(fold_best_vals)) + 1
    tprint("\n==== 10-FOLD CV SUMMARY ====")
    tprint(f"Validation loss across folds: {mean_val:.4f} ± {std_val:.4f}")
    tprint(f"Individual fold losses:")
    for i, v in enumerate(fold_best_vals, 1):
        tprint(f"  Fold {i}: {v:.4f}")
    tprint(f"Best fold: {best_fold} (val_total={min(fold_best_vals):.4f})")
    tprint(f"Best model saved to: {final_path}")


if __name__ == "__main__":
    main()
