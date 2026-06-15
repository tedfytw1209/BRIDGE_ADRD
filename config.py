config = {
    # Model parameters
    'graph_heads': 8,
    'theta': 0.0,
    'gcn_dim': 64,
    'top_k': 5,
    'num_gcn_layers': 2,


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
    'lambda_imp': 3.0,

    # Training parameters
    'dropout': 0.2,
    'epochs': 50,
    'lr': 0.0005,
    'weight_decay': 0.0001,
    'patience': 15,

    'test_size': 0.2,
    'batch_size': 64,
    'random_state': 42,

    'data_paths': {
        'obs_periods': 'Features_New/observation_periods.csv',
        'm1_features': 'Features_New/m1_features.csv',
        'm2_features': 'Features_New/m2_features.csv',
        'm3_features': 'Features_New/m3_features_clean.csv',
        'm4_features': 'Features_New/m4_features.csv',
        'm5_features': 'Features_New/m5_features_clean.csv',
        'm6_features': 'Features_New/m6_features.csv',
        'm7_features': 'Features_New/m7_features_old.csv',
    },

    # Feature dimensions
    'modality_dims': {'m1': 11, 'm2': 13, 'm3': 7, 'm4': 7, 'm5': 9, 'm6': 6, 'm7': 8,},
    'modalities_to_impute': ['m3', 'm5', 'm6', 'm7'],
    'categorical_modalities': ['m6'],
    'modalities_to_scale': ['m3', 'm7'],

    # Outlier clipping (train-only thresholds)
    # Features to clip within each modality (column indices)
    'outlier_clip': {
        'm7': {
            'percentile': 95,
            'features': ['Ttau', 'AB42', 'AB40', 'Ptau181', 'NFL', 'GFAP'],
        },
    },

    'save_model_path': 'output/best_model.pth',

    # Stage 2: Decoder finetuning hyperparameters (decoder_mod.py)
    'finetune_lr':         0.001,
    'finetune_epochs':     100,
    'finetune_patience':   20,
    'finetune_val_frac':   0.1,
    'finetune_model_path': 'output/best_model_finetuned.pth',

    # Ablation / runtime override flags
    'output_dir':     'output',
    'skip_graph':     0,
    'filter_graph':   1,
    'fusion_mode':    'gated',
    'lambda_cl':      1.0,
    'reliable_ratio': 1,
    'modality_a':     0,   # 1 = mask absent-modality patients from being GCN neighbors
}
