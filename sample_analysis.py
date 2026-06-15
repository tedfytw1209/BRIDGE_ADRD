"""Training sample count vs model MAE advantage over best baseline."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from data_loader import get_or_create_split

FEATURE_NAMES = {
    'm3': ['TotalChol', 'HDL', 'Glucose', 'HbA1c', 'BMI', 'SBP', 'DBP'],
    'm5': ['MMSE', 'MOCATOTS', 'CDRSUM', 'CDRGLOB', 'MINTTOTS', 'CRAFTURS', 'REY1REC', 'GDS', 'MOCA_CLOCK_TOTAL'],
    'm7': ['Ttau', 'AB42', 'AB40', 'AB4240ratio', 'Ptau181', 'NFL', 'GFAP', 'Ptau217'],
}

MODALITY_COLORS = {'m3': '#0051FF', 'm5': '#00AA00', 'm7': '#DD0000'}
MODALITY_MARKERS = {'m3': 'o', 'm5': 's', 'm7': 'D'}


def main():
    os.makedirs('plots', exist_ok=True)

    train_data_split, _, _ = get_or_create_split(config.config, save=True)

    df = pd.read_csv('output/imputation_all_modalities.csv')
    df['Method'] = df['Method'].str.strip()
    df['Feature'] = df['Feature'].str.strip()

    points = []
    for mod_key in ['m3', 'm5', 'm7']:
        train_data = train_data_split[mod_key]
        fnames = FEATURE_NAMES[mod_key]
        mod_upper = mod_key.upper()

        for j, fn in enumerate(fnames):
            train_obs = int(np.sum(~np.isnan(train_data[:, j])))

            rows = df[(df['Modality'] == mod_upper) & (df['Feature'] == fn)]
            model_mae = rows[rows['Method'] == 'Model']['MAE'].values
            baseline_maes = rows[rows['Method'] != 'Model']['MAE'].values

            if len(model_mae) == 0 or len(baseline_maes) == 0:
                continue
            if np.isnan(model_mae[0]) or np.all(np.isnan(baseline_maes)):
                continue

            best_baseline = np.nanmin(baseline_maes)
            if best_baseline == 0:
                continue
            # Positive = model is better, Negative = baseline wins
            improvement_pct = (best_baseline - model_mae[0]) / best_baseline * 100

            points.append({'feature': fn, 'modality': mod_key, 'train_obs': train_obs, 'improvement_pct': improvement_pct,})

    fig, ax = plt.subplots(figsize=(10, 6))
    for mod_key in ['m3', 'm5', 'm7']:
        mod_points = [p for p in points if p['modality'] == mod_key]
        x = [p['train_obs'] for p in mod_points]
        y = [p['improvement_pct'] for p in mod_points]
        labels_list = [p['feature'] for p in mod_points]

        ax.scatter(x, y, c=MODALITY_COLORS[mod_key], marker=MODALITY_MARKERS[mod_key], s=80, label=mod_key.upper(), edgecolors='black', linewidths=0.5, zorder=3)

        for xi, yi, lbl in zip(x, y, labels_list):
            ax.annotate(lbl, (xi, yi), textcoords='offset points', xytext=(6, 6), fontsize=7, alpha=0.85)

    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Shade regions
    ax.axhspan(0, ax.get_ylim()[1], color='green', alpha=0.05)
    ax.axhspan(ax.get_ylim()[0], 0, color='red', alpha=0.05)
    ax.text(0.98, 0.95, 'Model wins', transform=ax.transAxes, ha='right', va='top', fontsize=10, color='green', alpha=0.7, fontweight='bold')
    ax.text(0.98, 0.05, 'Baseline wins', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='red', alpha=0.7, fontweight='bold')

    ax.set_xlabel('Number of Observed Training Samples', fontsize=12)
    ax.set_ylabel('Model MAE Improvement over Best Baseline (%)', fontsize=12)
    ax.set_title('Training Sample Availability vs Model Advantage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = 'plots/sample_vs_performance.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
