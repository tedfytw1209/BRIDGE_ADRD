import numpy as np
import joblib
import os

from data_loader import _get_paths

# Load data
paths = _get_paths()
data = np.load(paths['split_path'], allow_pickle=True)
test_data = data["test_data"].item()
patient_ids = test_data["patient_ids"]

scalers_data = joblib.load(paths['scaler_path'])
scalers, medians = scalers_data["scalers"], scalers_data["medians"]

modalities = ["m3", "m5", "m6", "m7"]
methods = ["model", "mean", "median", "knn", "linearregression"]
method_labels = {"model": "Model", "mean": "Mean", "median": "Median", "knn": "KNN", "linearregression": "LinReg"}

feature_names = {
    "m3": ["TotalChol", "HDL", "Glucose", "HbA1c", "BMI", "SBP", "DBP"],
    "m5": ["MMSE", "MOCATOTS", "CDRSUM", "CDRGLOB", "MINTTOTS", "CRAFTURS", "REY1REC", "GDS", "MOCA_CLOCK_TOTAL"],
    "m6": ["APOE_22", "APOE_23", "APOE_24", "APOE_33", "APOE_34", "APOE_44"],
    "m7": ["Ttau", "AB42", "AB40", "AB42/40", "Ptau181", "NFL", "GFAP", "Ptau217"],
}

# Load imputation files
imputations = {}
for mod in modalities:
    imputations[mod] = {}
    for method in methods:
        path = f"imputations/test_{method}_{mod}.npy"
        if os.path.exists(path):
            imputations[mod][method] = np.load(path)

# sample patients
def pick_patients(mod_data, n_partial=3, n_complete=1):
    """Pick patients with partial and complete missingness."""
    nan_mask = np.isnan(mod_data)
    all_nan = np.all(nan_mask, axis=1)
    any_nan = np.any(nan_mask, axis=1) & ~all_nan
    partial_idx = np.where(any_nan)[0][:n_partial]
    complete_idx = np.where(all_nan)[0][:n_complete]
    return list(partial_idx) + list(complete_idx)

# Format value
def fmt(val):
    if np.isnan(val): return "NaN"
    if val == int(val): return f"{int(val)}"
    if round(val, 1) == val: return f"{val:.1f}"
    if abs(val) >= 1: return f"{val:.2f}"
    return f"{val:.4f}"

lines = []
lines.append("=" * 100)
lines.append("PATIENT-LEVEL IMPUTATION COMPARISON")
lines.append("Values marked with (*) were originally NaN and imputed by each method")
lines.append("=" * 100)

for mod in modalities:
    raw = test_data[mod]
    fnames = feature_names[mod]

    if mod == "m6":
        sample_idx = pick_patients(raw, n_partial=0, n_complete=3)
    else:
        sample_idx = pick_patients(raw, n_partial=3, n_complete=2)

    lines.append(f"\n{'-' * 100}")
    lines.append(f"  {mod.upper()} — {len(fnames)} features, {int(np.isnan(raw).mean() * 100)}% missing overall")
    lines.append(f"{'-' * 100}")

    for idx in sample_idx:
        pid = patient_ids[idx]
        original_row = raw[idx]
        nan_flags = np.isnan(original_row)
        n_missing = nan_flags.sum()
        status = "COMPLETELY MISSING" if nan_flags.all() else f"{n_missing}/{len(fnames)} missing"

        lines.append(f"\n  Patient {pid} (index {idx}) — {status}")
        lines.append(f"  {'─' * 90}")

        # Header
        col_w = max(12, max(len(fn) for fn in fnames) + 2)
        header = f"  {'Feature':<{col_w}} {'Original':>10}  "
        header += "  ".join(f"{method_labels[m]:>10}" for m in methods if m in imputations[mod])
        lines.append(header)
        lines.append(f"  {'─' * (len(header) - 2)}")

        for j, fn in enumerate(fnames):
            orig_val = original_row[j]
            is_nan = np.isnan(orig_val)
            marker = " (*)" if is_nan else "    "

            row_str = f"  {fn:<{col_w}} {'NaN':>10}{marker}" if is_nan else f"  {fn:<{col_w}} {fmt(orig_val):>10}    "

            for method in methods:
                if method not in imputations[mod]:
                    continue
                imp_val = imputations[mod][method][idx, j]
                row_str += f"  {fmt(imp_val):>10}"

            lines.append(row_str)

        # For M6: show predicted genotype
        if mod == "m6" and nan_flags.all():
            lines.append(f"\n  Predicted genotype:")
            for method in methods:
                if method not in imputations[mod]:
                    continue
                pred_row = imputations[mod][method][idx]
                pred_class = np.argmax(pred_row)
                genotypes = ["22", "23", "24", "33", "34", "44"]
                confidence = "one-hot" if pred_row[pred_class] == 1.0 else f"max={pred_row[pred_class]:.3f}"
                lines.append(f"    {method_labels[method]:>10}: APOE {genotypes[pred_class]} ({confidence})")

output = "\n".join(lines)
print(output)

# Save
with open("output/imputation_patient_comparison.txt", "w") as f:
    f.write(output)
print(f"\n\nSaved to output/imputation_patient_comparison.txt")