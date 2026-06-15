config = {
    # Model parameters
    'graph_heads': 8,
    'theta': 0.0,
    'gcn_dim': 16,
    'top_k': 5,
    'filter_graph': True,
    'num_gcn_layers': 1,


    # Data augmentation probabilities
    'prob_m1': 0.1,
    'prob_m2': 0.1,
    'prob_m3': 0.2,
    'prob_m4': 0.1,
    'prob_m5': 0.15,
    'prob_m6': 0.05,
    'prob_m7': 0.25,
    'prob_edge': 0.4,

    # Evaluation masking probabilities (Table 2a: 30% random dropout)
    'eval_prob_m3': 0.3,
    'eval_prob_m5': 0.3,
    'eval_prob_m6': 0.3,
    'eval_prob_m7': 0.3,

    # Contrastive learning parameters
    'tau': 0.5,
    'alpha1': 0.55, #(z as anchor)
    'alpha2': 0.45, #(z_tilde as anchor)
    'lambda_cl': 0,
    'lambda_imp': 2.0,
    'lambda_recon': 0,

    # Training parameters
    'dropout': 0.2,
    'epochs': 100,
    'lr': 0.0005,
    'weight_decay': 1e-3,
    'patience': 15,
    # LR scheduler: 'none', 'cosine', 'step'
    'scheduler': 'none',
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    # Early stopping metric — any key from val_metrics or imp_eval_metrics (lower=better).
    # Options: 'val_total_loss', 'val_imputation_loss', 'val_contrastive_loss',
    #          'val_reconstruction_loss', 'val_imp_eval_total_mean_mape',
    #          'val_imp_eval_m3_mean_mape', 'val_imp_eval_m5_mean_mape',
    #          'val_imp_eval_m7_mean_mape', 'val_imp_eval_m6_accuracy' (use negation not yet supported)
    'early_stop_metric': 'val_imp_eval_total_mean_mape',

    'test_size': 0.2,
    'batch_size': 64,
    'random_state': 42,

    'data_paths': {
        'obs_periods': 'Features_New/observation_periods.csv',
        'm1_features': 'Features_New/m1_features.csv',
        'm2_features': 'Features_New/m2_features.csv',
        'm3_features': 'Features_New/m3_features_clean.csv',
        'm4_features': 'Features_New/m4_features.csv',
        'm5_features': 'Features_New/M5_features_scaled.csv', # 2/28 tmp change
        'm6_features': 'Features_New/m6_features.csv',
        'm7_features': 'Features_New/M7_features.csv',
    },

    # Feature dimensions
    'modality_dims': {'m1': 11, 'm2': 13, 'm3': 7, 'm4': 7, 'm5': 9, 'm6': 6, 'm7': 8,},
    'modalities_to_impute': ['m3', 'm5', 'm6', 'm7'],
    'categorical_modalities': ['m6'],
    #'sigmoid_modalities': ['m5'],
    'sigmoid_modalities': [],
    'modalities_to_scale': ['m3', 'm7'],
    #'modalities_to_scale': ['m3', 'm5', 'm7'],
    
    # Build-up experiment flags
    "skip_graph": False,       # True = encoder-decoder only (no graph/GCN/fusion)
    "fusion_mode": "gated",    # "gated" (learned gates) or "average" (simple 0.5/0.5)
    "reliable_ratio": True,  # If True, use overall reliable ratio for weighting

    # Propensity-weighted graph construction (Option B: causal bias reduction)
    # Mode: "none" = disabled, "ipw" = inverse propensity weighting, "caliper" = caliper pruning
    "propensity_reweight": "none",
    # Source: "all" = all modalities as covariates (NaN->0), "obs" = always-observed only (m1, m2, m4)
    "propensity_source": "obs",
    "propensity_caliper_mult": 0.2,    # caliper = mult * SD(logit(p_hat))
    "propensity_clip_min": 0.05,       # clip propensity scores to [min, max]
    "propensity_clip_max": 0.95,

    #GCN new parameters
    "modality_wise_A": True,
    "modality_wise_fusion": False,
    "gcn_use_residual": True,
    "gcn_use_layernorm": False,
    "gcn_norm_position": "pre",
    
    #Encoder and decoder parameters
    "encoder_type": "linear",  # "linear" or "mlp2"
    "encoder_act": "gelu",  # "relu" or "gelu"
    "encoder_dropout": 0.1,
    "encoder_use_layernorm": False,
    "decoder_type": "linear",  # "linear", "mlp2", or "sigmoid_mlp"
    "decoder_act": "gelu",  # "relu" or "gelu"
    "decoder_dropout": 0.1,
    "decoder_use_layernorm": False,

    # Stage 2: Decoder finetuning hyperparameters (decoder_mod.py)
    'finetune_lr':         0.002,
    'finetune_epochs':     100,
    'finetune_patience':   20,
    'finetune_val_frac':   0.1,
    'finetune_model_path': 'best_model_finetuned.pth',
    'save_model_name': 'best_model.pth',

    # Stage 2 v2: Feature-wise decoder finetuning (decoder_mod_v2.py)
    'feat_finetune_lr':           0.002,
    'feat_finetune_epochs':       100,
    'feat_finetune_patience':     20,
    'feat_finetune_val_frac':     0.1,
    'feat_finetune_model_path':   'best_model_feat_finetuned.pth',
    # LR scheduler for stage 2 v2: 'none', 'cosine', 'step'
    'feat_finetune_scheduler':    'cosine',
    'feat_finetune_warmup_epochs': 5,
    'feat_finetune_min_lr':        5e-4,
}

# Apply overrides from environment (used by sweep / run_pipeline subprocesses)
import os as _os
import json as _json

_override_path = _os.environ.get('PSN_CONFIG_OVERRIDE')
if _override_path and _os.path.exists(_override_path):
    with open(_override_path) as _f:
        config.update(_json.load(_f))

_output_dir = _os.environ.get('PSN_OUTPUT_DIR')
if _output_dir:
    config['save_model_path'] = _os.path.join(_output_dir, config['save_model_name'])
