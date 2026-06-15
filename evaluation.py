import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from utils import compute_all_metrics


# Plotting
METHOD_COLORS = {'Model': '#2E86AB', 'Mean': '#A23B72', 'Median': '#6A4C93', 'KNN': '#F18F01', 'KNN1': '#F5A623', 'KNN3': '#F7C948', 'LinearRegression': '#C73E1D', 'GAIN': '#3BB273', 'GRAPE': '#2EC4B6'}


def plot_bars_by_feature(metric_table, feat_names, outfile, title):

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')

    df_plot = pd.DataFrame(metric_table, index=feat_names).reset_index().melt(id_vars='index', var_name='Method', value_name='Metric').rename(columns={'index': 'Feature'})

    plt.figure(figsize=(max(8, len(feat_names) * 1.2), 6))
    ax = sns.barplot(data=df_plot, x='Feature', y='Metric', hue='Method', palette=METHOD_COLORS, edgecolor='black', alpha=0.85)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(title.split(' ')[0].strip()+" Error", fontsize=12, fontweight='bold')
    plt.xlabel('Feature', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Method', title_fontsize=11, fontsize=10, loc='upper right', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ax.set_facecolor('#F8F9FA')

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_single_feature(methods, values, outfile, title, metric_name, n=None):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')

    colors = [METHOD_COLORS.get(m, '#808080') for m in methods]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, values, color=colors, edgecolor='black', alpha=0.85, width=0.6)

    # Add value labels on top of bars
    max_val = max([v for v in values if np.isfinite(v)], default=1.0)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * max_val,f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plot_title = f"{title}  (n={n})" if n is not None else title
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.ylabel(f'{metric_name} Error', fontsize=12, fontweight='bold')
    plt.xlabel('Method', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gca().set_facecolor('#F8F9FA')
    plt.tight_layout()

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_categorical_metrics(mod_key, methods_to_eval, cat_results, save_dir="plots"):
    os.makedirs(os.path.join(save_dir, mod_key), exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')
    colors = [METHOD_COLORS.get(m, '#808080') for m in methods_to_eval]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric_key, metric_label in zip(axes, ['accuracy', 'auroc'], ['Accuracy', 'AUROC']):
        values = [cat_results[m][metric_key] for m in methods_to_eval]

        # Baselines have no probability scores for categorical modalities, so their AUROC is NaN.
        valid = [(m, v, c) for m, v, c in zip(methods_to_eval, values, colors) if np.isfinite(v)]

        ax.set_title(f'{mod_key.upper()} - {metric_label}', fontsize=14, fontweight='bold')
        ax.set_facecolor('#F8F9FA')

        if not valid:
            ax.text(0.5, 0.5, f'{metric_label} not available\n(no probability scores for these methods)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray', style='italic')
            continue

        valid_methods, valid_values, valid_colors = zip(*valid)
        bars = ax.bar(valid_methods, valid_values, color=list(valid_colors), edgecolor='black', alpha=0.85, width=0.6)
        for bar, val in zip(bars, valid_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(valid_methods)))
        ax.set_xticklabels(valid_methods, rotation=45, ha='right', fontsize=10)
        ax.set_ylim([0, 1.15])
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_key, "classification_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_classification_results(classification_results, save_dir="plots/"):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    classifier_styles = {'Logistic Regression': '-', 'Random Forest': '--', 'XGBoost': '-.'}

    # ========== ROC Curves ==========
    ax_roc = fig.add_subplot(gs[0, :2])

    # Find the best classifier per imputation method (by AUROC)
    best_clf_per_method = {}
    for key, results in classification_results.items():
        method_name = key.split(' - ')[0]
        auroc = results["metrics"]["auroc_ovr"]
        if method_name not in best_clf_per_method or auroc > best_clf_per_method[method_name][1]:
            best_clf_per_method[method_name] = (key, auroc)

    best_keys = {v[0] for v in best_clf_per_method.values()}

    for key, results in classification_results.items():
        method_name = key.split(' - ')[0]
        classifier_name = ' - '.join(key.split(' - ')[1:])

        y_true = results["y_true"]
        y_prob = results["y_prob"]

        class_idx = 1  # MCI
        y_true_binary = (y_true == class_idx).astype(int)
        y_prob_class = y_prob[:, class_idx]

        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        auc = roc_auc_score(y_true_binary, y_prob_class)
        color = METHOD_COLORS.get(method_name, '#333333')
        linestyle = classifier_styles.get(classifier_name, '-')

        # Highlight the best classifier per imputation method
        if key in best_keys:
            label = f"{method_name} - {classifier_name} (AUC={auc:.3f})"
            linewidth = 2.5
            alpha = 1.0
        else:
            label = None
            linewidth = 1.5
            alpha = 0.3

        ax_roc.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_title('ROC Curves - MCI vs. Rest (Best Classifier per Method)', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    ax_roc.grid(True, alpha=0.3, linestyle='--')
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_facecolor('#F8F9FA')

    # ========== Confusion Matrix ==========
    ax_cm = fig.add_subplot(gs[0, 2])

    # Get overall best model by AUROC
    best_key = max(classification_results, key=lambda k: classification_results[k]["metrics"]["auroc_ovr"])
    best_results = classification_results[best_key]
    cm = confusion_matrix(best_results["y_true"], best_results["y_pred"])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annotations, fmt='', ax=ax_cm, cmap='Blues', xticklabels=['Normal', 'MCI', 'Dementia'],
                yticklabels=['Normal', 'MCI', 'Dementia'], cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')

    ax_cm.set_title(f'Confusion Matrix\n{best_key}', fontsize=14, fontweight='bold')
    ax_cm.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # ========== Metrics Comparison ==========
    ax_metrics = fig.add_subplot(gs[1, :])

    metrics_data = []
    for key, results in classification_results.items():
        method_name = key.split(' - ')[0]
        classifier_name = ' - '.join(key.split(' - ')[1:])
        metrics = results["metrics"]

        metrics_data.append({
            'Method': method_name, 'Classifier': classifier_name, 'Accuracy': metrics['accuracy'], 'Precision': metrics['precision_macro'],
            'Recall': metrics['recall_macro'], 'F1-Score': metrics['f1_macro'], 'AUROC': metrics['auroc_ovr']
        })

    df_metrics = pd.DataFrame(metrics_data)

    # Best model by AUROC
    best_idx = df_metrics['AUROC'].idxmax()
    best_method = df_metrics.iloc[best_idx]['Method']
    best_classifier = df_metrics.iloc[best_idx]['Classifier']
    best_auroc = df_metrics.iloc[best_idx]['AUROC']

    print(f"\n[INFO] Best performer: {best_method} - {best_classifier} (AUROC={best_auroc:.3f})")

    x_labels = [f"{row['Method']}\n{row['Classifier']}" for _, row in df_metrics.iterrows()]
    x = np.arange(len(x_labels))
    width = 0.15

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    colors_metrics = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - len(metrics_to_plot) / 2 + 0.5)
        bars = ax_metrics.bar(x + offset, df_metrics[metric], width, label=metric, color=colors_metrics[i], alpha=0.8)

        for j, bar in enumerate(bars):
            if df_metrics.iloc[j]['Method'] == best_method and df_metrics.iloc[j]['Classifier'] == best_classifier:
                height = bar.get_height()
                ax_metrics.text(
                    bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold', color=colors_metrics[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors_metrics[i], alpha=0.8, linewidth=1)
                )

    ax_metrics.set_xlabel('Imputation Method - Classifier', fontsize=12, fontweight='bold')
    ax_metrics.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax_metrics.set_title('Performance Metrics Comparison Across All Combinations', fontsize=14, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax_metrics.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, frameon=True, shadow=True)
    ax_metrics.set_ylim([0, 1.05])
    ax_metrics.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_metrics.axhline(y=0.8, color='red', linestyle=':', linewidth=1, alpha=0.5, label='80% Threshold')
    ax_metrics.set_facecolor('#F8F9FA')

    fig.suptitle('Imputation-Based Classification Results (3-Class AD Staging)', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(f"{save_dir}/classification_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification plots saved to {save_dir}classification_results.png")

def evaluate_categorical(mod_key, true_std, true_original, pred_original, baseline_original, R_eval, feat_names, methods_to_eval, model_scores=None, baseline_scores=None):
    patient_masked = R_eval.sum(axis=1) > 0
    n_masked = int(patient_masked.sum())
    n_with_data = int(np.any(~np.isnan(true_std), axis=1).sum())
    print(f"Masked patients: {n_masked} ({100 * n_masked / max(n_with_data, 1):.1f}% of patients with data)")

    if n_masked == 0:
        print(f"  No masked patients for {mod_key.upper()}, skipping evaluation")
        return None

    true_classes = true_original[patient_masked].argmax(axis=1)

    cat_results = {}
    print(f"\n{'Method':<18} {'Accuracy':>10} {'AUROC':>10}")
    print("-" * 40)

    for method_name in methods_to_eval:
        if method_name == "Model":
            pred_data = pred_original
            scores = model_scores
        else:
            pred_data = baseline_original[method_name]
            scores = baseline_scores.get(method_name) if baseline_scores else None

        pred_classes = pred_data[patient_masked].argmax(axis=1)
        acc = accuracy_score(true_classes, pred_classes)

        # Compute AUROC using probability scores
        auroc = float('nan')
        if scores is not None:
            scores_masked = scores[patient_masked]
            # Normalize to probabilities (clip negatives, normalize rows to sum to 1)
            scores_clipped = np.clip(scores_masked, 0, None)
            row_sums = scores_clipped.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            scores_prob = scores_clipped / row_sums
            try:
                auroc = roc_auc_score(true_classes, scores_prob, multi_class='ovr', average='macro')
            except ValueError:
                auroc = float('nan')

        cat_results[method_name] = {'accuracy': acc, 'auroc': auroc}
        print(f"{method_name:<18} {acc:>10.4f} {auroc:>10.4f}")

    return cat_results


def evaluate_continuous(mod_key, true_original, pred_original, baseline_original, R_eval, feat_names, methods_to_eval, patient_ids):
    idx = R_eval.astype(bool)
    print(f"Hidden entries: {int(idx.sum())} ({100 * idx.sum() / np.isfinite(true_original).sum():.1f}% of observed)")

    header = f"{'Method':<18} {'Feature':<17} {'MAE':>10} {'MSE': >12} {'RMSE':>14} {'MAPE':>12} {'R2':>10}"
    print("\n" + header)
    print("-" * len(header))

    perfeat_results = {method: {'mae': [], 'mse': [], 'rmse': [],  'mape': [], 'r2': []} for method in methods_to_eval}
    feat_n = []  # number of evaluation samples per feature
    for j, feat_name in enumerate(feat_names):
        mask_j = idx[:, j]
        if mask_j.sum() == 0:
            feat_n.append(0)
            for method in methods_to_eval:
                perfeat_results[method]['mae'].append(np.nan)
                perfeat_results[method]['mse'].append(np.nan)
                perfeat_results[method]['rmse'].append(np.nan)
                perfeat_results[method]['mape'].append(np.nan)
                perfeat_results[method]['r2'].append(np.nan)
            continue

        y_true = true_original[:, j][mask_j]
        feat_n.append(int(np.sum(np.isfinite(y_true))))

        # Model
        y_pred_model = pred_original[:, j][mask_j]
        m_metrics = compute_all_metrics(y_true, y_pred_model)
        for k, v in m_metrics.items():
            perfeat_results["Model"][k].append(v)

        print(f"{'Model':<18} {feat_name:<17} {m_metrics['mae']:>10.4f} {m_metrics['mse']:>13.4f} {m_metrics['rmse']:>12.4f} {m_metrics['mape']:>14.2f}% {m_metrics['r2']:>10.4f}")

        # Baselines
        for method in baseline_original.keys():
            y_pred_baseline = baseline_original[method][:, j][mask_j]
            b_metrics = compute_all_metrics(y_true, y_pred_baseline)
            for k, v in b_metrics.items():
                perfeat_results[method][k].append(v)

            print(f"{method:<18} {feat_name:<17} {b_metrics['mae']:>10.4f} {b_metrics['mse']:>13.4f} {b_metrics['rmse']:>12.4f} {b_metrics['mape']:>14.2f}% {b_metrics['r2']:>10.4f}")

    # Per-patient aggregated MAE
    patient_mae_results = {method: [] for method in methods_to_eval}

    for i in range(len(patient_ids)):
        hidden_i = idx[i]
        if hidden_i.sum() == 0: continue
        true_i = true_original[i, hidden_i]

        # Model
        pred_i_model = pred_original[i, hidden_i]
        patient_mae_results["Model"].append(np.mean(np.abs(true_i - pred_i_model)))
        for method in baseline_original.keys():
            pred_i_baseline = baseline_original[method][i, hidden_i]
            patient_mae_results[method].append(np.mean(np.abs(true_i - pred_i_baseline)))

    print(f"\n{'Method':<18} {'Mean Patient MAE':>20}")
    print("-" * 40)
    for method in methods_to_eval:
        print(f"{method:<18} {np.mean(patient_mae_results[method]):>20.4f}")

    patient_mae = {method: np.mean(patient_mae_results[method]) if len(patient_mae_results[method]) > 0 else np.nan for method in methods_to_eval}

    return perfeat_results, patient_mae, feat_n


def evaluate_continuous_trimmed(mod_key, true_original, pred_original, baseline_original, R_eval,
                                feat_names, methods_to_eval, patient_ids, trim_percentile=95):
    """Same as evaluate_continuous but excludes top outliers per feature from ground truth.
    For each feature, samples whose ground truth exceeds the trim_percentile-th value are excluded."""
    idx = R_eval.astype(bool)

    header = f"{'Method':<18} {'Feature':<17} {'MAE':>10} {'MSE': >12} {'RMSE':>14} {'MAPE':>12} {'R2':>10}"
    print(f"\n[Trimmed p{trim_percentile}] Excluding top {100 - trim_percentile}% outliers per feature")
    print(header)
    print("-" * len(header))

    perfeat_results = {method: {'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'r2': []} for method in methods_to_eval}
    feat_n = []
    for j, feat_name in enumerate(feat_names):
        mask_j = idx[:, j]
        if mask_j.sum() == 0:
            feat_n.append(0)
            for method in methods_to_eval:
                for k in ['mae', 'mse', 'rmse', 'mape', 'r2']:
                    perfeat_results[method][k].append(np.nan)
            continue

        y_true_all = true_original[:, j][mask_j]
        finite_mask = np.isfinite(y_true_all)
        if finite_mask.sum() == 0:
            feat_n.append(0)
            for method in methods_to_eval:
                for k in ['mae', 'mse', 'rmse', 'mape', 'r2']:
                    perfeat_results[method][k].append(np.nan)
            continue

        threshold = np.percentile(y_true_all[finite_mask], trim_percentile)
        keep = y_true_all <= threshold
        y_true = y_true_all[keep]
        feat_n.append(int(np.sum(np.isfinite(y_true))))

        y_pred_model = pred_original[:, j][mask_j][keep]
        m_metrics = compute_all_metrics(y_true, y_pred_model)
        for k, v in m_metrics.items():
            perfeat_results["Model"][k].append(v)
        print(f"{'Model':<18} {feat_name:<17} {m_metrics['mae']:>10.4f} {m_metrics['mse']:>13.4f} {m_metrics['rmse']:>12.4f} {m_metrics['mape']:>14.2f}% {m_metrics['r2']:>10.4f}")

        for method in baseline_original.keys():
            y_pred_bl = baseline_original[method][:, j][mask_j][keep]
            b_metrics = compute_all_metrics(y_true, y_pred_bl)
            for k, v in b_metrics.items():
                perfeat_results[method][k].append(v)
            print(f"{method:<18} {feat_name:<17} {b_metrics['mae']:>10.4f} {b_metrics['mse']:>13.4f} {b_metrics['rmse']:>12.4f} {b_metrics['mape']:>14.2f}% {b_metrics['r2']:>10.4f}")

    # Per-patient aggregated MAE (trimmed)
    patient_mae_results = {method: [] for method in methods_to_eval}
    for i in range(len(patient_ids)):
        hidden_i = idx[i]
        if hidden_i.sum() == 0:
            continue
        true_i = true_original[i, hidden_i]
        # Per-feature threshold: only keep features where this patient's value <= p95
        keep_feats = np.ones(hidden_i.sum(), dtype=bool)
        feat_indices = np.where(hidden_i)[0]
        for local_j, global_j in enumerate(feat_indices):
            col_vals = true_original[:, global_j][idx[:, global_j]]
            finite_vals = col_vals[np.isfinite(col_vals)]
            if len(finite_vals) > 0:
                thresh = np.percentile(finite_vals, trim_percentile)
                if not np.isfinite(true_i[local_j]) or true_i[local_j] > thresh:
                    keep_feats[local_j] = False
        if keep_feats.sum() == 0:
            continue
        true_kept = true_i[keep_feats]
        pred_model_kept = pred_original[i, hidden_i][keep_feats]
        patient_mae_results["Model"].append(np.mean(np.abs(true_kept - pred_model_kept)))
        for method in baseline_original.keys():
            pred_bl_kept = baseline_original[method][i, hidden_i][keep_feats]
            patient_mae_results[method].append(np.mean(np.abs(true_kept - pred_bl_kept)))

    print(f"\n{'Method':<18} {'Mean Patient MAE':>20}")
    print("-" * 40)
    for method in methods_to_eval:
        val = np.mean(patient_mae_results[method]) if len(patient_mae_results[method]) > 0 else np.nan
        print(f"{method:<18} {val:>20.4f}")

    patient_mae = {method: np.mean(patient_mae_results[method]) if len(patient_mae_results[method]) > 0 else np.nan for method in methods_to_eval}
    return perfeat_results, patient_mae, feat_n


def save_classification_csv(classification_results, csv_path):
    rows = []
    for key, result in classification_results.items():
        parts = key.split(' - ')
        imp_method = parts[0]
        classifier = ' - '.join(parts[1:])
        m = result["metrics"]
        rows.append({
            'ImpMethod': imp_method,
            'Classifier': classifier,
            'Accuracy': round(m['accuracy'], 4),
            'Precision_Macro': round(m['precision_macro'], 4),
            'Recall_Macro': round(m['recall_macro'], 4),
            'F1_Macro': round(m['f1_macro'], 4),
            'Precision_Weighted': round(m['precision_weighted'], 4),
            'Recall_Weighted': round(m['recall_weighted'], 4),
            'F1_Weighted': round(m['f1_weighted'], 4),
            'AUROC': round(m['auroc_ovr'], 4),
            'AUPRC': round(m['auprc'], 4),
        })

    rows.sort(key=lambda r: r['AUROC'], reverse=True)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved classification results to {csv_path}")


def save_results_csv(all_results, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Modality", "Type", "Method", "Feature", "MAE", "MSE", "RMSE", "MAPE", "R2", "Accuracy", "AUROC"])
        for mod_key, res in all_results.items():
            if res.get('type') == 'categorical':
                for method_name, metrics in res['cat_results'].items():
                    auroc_val = round(metrics['auroc'], 4) if np.isfinite(metrics['auroc']) else "N/A"
                    w.writerow([mod_key.upper(), "categorical", method_name, "ALL", "N/A", "N/A", "N/A", "N/A", "N/A", round(metrics['accuracy'], 4), auroc_val])
            else:
                for method, metrics in res['perfeat_results'].items():
                    for feat, vals in zip(res['feat_names'], zip(metrics['mae'], metrics['mse'], metrics['rmse'], metrics['mape'], metrics['r2'])):
                        w.writerow([mod_key.upper(), "continuous", method, feat, *[round(v, 4) if np.isfinite(v) else "N/A" for v in vals], "N/A", "N/A"])
    print(f"Saved combined results to {csv_path}")
