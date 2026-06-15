import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import config
from data_loader import get_or_create_split, get_or_create_eval_mask


FEATURE_NAMES = {
    'm3': ['TotalChol', 'HDL', 'Glucose', 'HbA1c', 'BMI', 'SBP', 'DBP'],
    'm5': ['MMSE', 'MOCATOTS', 'CDRSUM', 'CDRGLOB', 'MINTTOTS', 'CRAFTURS', 'REY1REC', 'GDS', 'MOCA_CLOCK_TOTAL'],
    'm7': ['Ttau', 'AB42', 'AB40', 'AB4240ratio', 'Ptau181', 'NFL', 'GFAP', 'Ptau217'],
}

METHODS = {
    'model': {'label': 'Model',  'color': '#0051FF', 'marker': 'o'},
    'knn': {'label': 'KNN',    'color': '#FF6600', 'marker': 's'},
    'mean': {'label': 'Mean',   'color': '#00AA00', 'marker': '^'},
    'median': {'label': 'Median', 'color': '#9933CC', 'marker': 'v'},
    'linearregression': {'label': 'LinReg', 'color': '#DD0000', 'marker': 'D'},
    'gain': {'label': 'GAIN',   'color': '#3BB273', 'marker': '*'},
    'grape': {'label': 'GRAPE',  'color': '#2EC4B6', 'marker': 'P'},
}

# Suppress LAPACK warnings
_old_stderr = None
def _suppress_stderr():
    global _old_stderr
    sys.stderr.flush()
    _old_stderr = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

def _restore_stderr():
    global _old_stderr
    sys.stderr.flush()
    os.dup2(_old_stderr, 2)
    os.close(_old_stderr)


def load_test_ground_truth():
    _, test_data, _ = get_or_create_split(config.config, save=True)
    return test_data


def scatter_plot(true_vals, errors, method_info, ax):
    """Ground truth (x) vs absolute error (y), one panel per method."""
    valid = np.isfinite(true_vals) & np.isfinite(errors)
    true_vals = true_vals[valid]
    errors = errors[valid]

    ax.scatter(true_vals, errors, alpha=0.7, s=22, c=method_info['color'], marker=method_info['marker'], edgecolors='none')

    mae = np.mean(errors) if len(errors) > 0 else 0.0
    ax.axhline(mae, color=method_info['color'], linestyle='--', linewidth=1.0, alpha=0.8)

    if len(true_vals) > 5 and np.std(true_vals) > 1e-12:
        _suppress_stderr()
        z = np.polyfit(true_vals, errors, 1)
        _restore_stderr()
        x_sorted = np.sort(true_vals)
        ax.plot(x_sorted, np.polyval(z, x_sorted), color='black', linewidth=1.2, alpha=0.6)

    ax.set_title(f'{method_info["label"]}  (MAE={mae:.3f}, n={len(true_vals)})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Ground Truth', fontsize=9)
    ax.set_ylabel('Absolute Error', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def generate_feature_plot(mod_key, feat_idx, feat_name, true_original, R_eval, out_dir, imp_dir,
                          trim_percentile=None):
    """Generate scatter plots. If trim_percentile is set, exclude samples above that percentile."""
    mask_col = R_eval[:, feat_idx]
    gt_vals = true_original[:, feat_idx][mask_col]

    if trim_percentile is not None:
        finite = gt_vals[np.isfinite(gt_vals)]
        if len(finite) > 0:
            threshold = np.percentile(finite, trim_percentile)
            keep = gt_vals <= threshold
        else:
            keep = np.ones(len(gt_vals), dtype=bool)
    else:
        keep = np.ones(len(gt_vals), dtype=bool)

    gt_vals_plot = gt_vals[keep]

    n_methods = len(METHODS)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=True)
    for ax in axes.flat[n_methods:]:
        ax.set_visible(False)
    for ax, (method_key, minfo) in zip(axes.flat, METHODS.items()):
        pred_all = np.load(os.path.join(imp_dir, f"test_{method_key}_{mod_key}.npy"))
        pred_vals = pred_all[:, feat_idx][mask_col][keep]
        abs_errors = np.abs(pred_vals - gt_vals_plot)
        scatter_plot(gt_vals_plot, abs_errors, minfo, ax)

    suffix = f" (Trimmed p{trim_percentile})" if trim_percentile else ""
    fig.suptitle(f'{mod_key.upper()} - {feat_name}: Ground Truth vs Absolute Error{suffix}',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    tag = f"_trimmed_p{trim_percentile}" if trim_percentile else ""
    out_path = os.path.join(out_dir, f'{feat_name}_error_scatter{tag}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_and_override(cfg):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',   type=str, default=None)
    p.add_argument('--random-state', type=int, default=None)
    args, _ = p.parse_known_args()

    if args.output_dir   is not None: cfg['output_dir']   = args.output_dir
    if args.random_state is not None: cfg['random_state'] = args.random_state

    out = cfg['output_dir']
    cfg['save_model_path']     = os.path.join(out, 'output', 'best_model.pth')
    cfg['finetune_model_path'] = os.path.join(out, 'output', 'best_model_finetuned.pth')
    return args


def main():
    parse_and_override(config.config)
    OUT = config.config['output_dir']
    imp_dir  = os.path.join(OUT, 'imputations')
    plot_dir = os.path.join(OUT, 'plots', 'error_analysis')
    os.makedirs(plot_dir, exist_ok=True)
    seed = config.config['random_state']

    print("Loading test data")
    test_data = load_test_ground_truth()

    modalities = config.config['modalities_to_impute']
    categorical_mods = set(config.config['categorical_modalities'])

    for mod_key in modalities:
        if mod_key in categorical_mods:
            print(f"\nSkipping {mod_key.upper()} (categorical)")
            continue

        feat_names = FEATURE_NAMES.get(mod_key, [f'F{i}' for i in range(test_data[mod_key].shape[1])])
        feat_names = feat_names[:test_data[mod_key].shape[1]]

        print(f"\n{mod_key.upper()} - {len(feat_names)} features")
        print(f"{'='*60}")

        original_key = f'{mod_key}_original'
        if original_key in test_data:
            true_original = test_data[original_key].copy()
            print(f"  Using unclipped original values as ground truth for {mod_key.upper()}")
        else:
            true_original = test_data[mod_key].copy()
        mask_prob = config.config.get(f'eval_prob_{mod_key}', 0.2)
        R_eval = get_or_create_eval_mask(true_original, modality_key=mod_key, mask_prob=mask_prob, seed=seed)

        mod_out_dir = os.path.join(plot_dir, mod_key)
        os.makedirs(mod_out_dir, exist_ok=True)

        for j, feat_name in enumerate(feat_names):
            generate_feature_plot(mod_key, j, feat_name, true_original, R_eval, mod_out_dir, imp_dir)

        # Trimmed scatter plots for modalities with outlier clipping (e.g. M7)
        outlier_clip_mods = set(config.config.get('outlier_clip', {}).keys())
        if mod_key in outlier_clip_mods:
            trim_pct = 95
            trimmed_out_dir = os.path.join(plot_dir, f"{mod_key}_trimmed")
            os.makedirs(trimmed_out_dir, exist_ok=True)
            print(f"\n  Generating trimmed scatter plots (excluding top {100 - trim_pct}% outliers)")
            for j, feat_name in enumerate(feat_names):
                generate_feature_plot(mod_key, j, feat_name, true_original, R_eval,
                                      trimmed_out_dir, imp_dir, trim_percentile=trim_pct)


if __name__ == '__main__':
    main()
