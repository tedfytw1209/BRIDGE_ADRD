import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [1, 2, 3, 4, 42]
ALL_STEPS = ["train", "decoder_mod", "imputation", "error_analysis", "classification", "eval_demographics"]
RESULTS_DIR = "results_multi_seed"
DONE_MARKERS = [
    "output/classification_all_methods.csv", "output/imputation_all_modalities.csv", "output/best_model.pth",
    "output/best_model_finetuned.pth",
    "output/classification_by_demographics.csv",
    "imputations/test_model_m3.npy", "imputations/test_model_m5.npy", "imputations/test_model_m6.npy", "imputations/test_model_m7.npy",
    "imputations/test_gain_m3.npy", "imputations/test_gain_m5.npy", "imputations/test_gain_m7.npy",
    "imputations/test_grape_m3.npy", "imputations/test_grape_m5.npy", "imputations/test_grape_m7.npy",
    "imputations/test_mice_m3.npy", "imputations/test_missforest_m3.npy", "imputations/test_ppm_m3.npy",
    "embeddings/train_embeddings.npy", "embeddings/test_embeddings.npy",
]


def run_command(command):
    process = subprocess.Popen(command, shell=True, universal_newlines=True)
    process.wait()
    if process.returncode != 0:
        print(f"  {command.split()[1]} failed (exit {process.returncode})")
        return False
    return True


def run_one_seed(seed, seed_dir, steps_to_run=None):
    os.makedirs(seed_dir, exist_ok=True)

    out_dir_arg = f"--output-dir {seed_dir}"
    seed_arg    = f"--random-state {seed}"
    base        = sys.executable

    all_steps = [
        (f"{base} train.py {out_dir_arg} {seed_arg}",          True),
        (f"{base} decoder_mod.py {out_dir_arg} {seed_arg}",    True),
        (f"{base} imputation.py {out_dir_arg} {seed_arg}",     True),
        (f"{base} error_analysis.py {out_dir_arg} {seed_arg}", False),
        (f"{base} classification.py {out_dir_arg} {seed_arg}", True),
        (f"{base} eval_demographics.py {out_dir_arg} {seed_arg}", False),
    ]
    step_names = ["train", "decoder_mod", "imputation", "error_analysis", "classification", "eval_demographics"]

    steps = [(cmd, crit) for (cmd, crit), name in zip(all_steps, step_names)
             if steps_to_run is None or name in steps_to_run]

    for cmd, critical in steps:
        script = cmd.split()[1]
        print(f"  {script}")
        ok = run_command(cmd)
        if not ok and critical:
            return False

    return True


def compute_ci(values):
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = arr.mean()
    if n == 1:
        return mean, np.nan, np.nan, 1
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(arr))
    return mean, lo, hi, n


def aggregate_classification(seed_dirs):
    all_dfs = []
    for sd in seed_dirs:
        csv_path = os.path.join(sd, "output", "classification_all_methods.csv")
        if os.path.exists(csv_path):
            all_dfs.append(pd.read_csv(csv_path))

    if len(all_dfs) < len(SEEDS):
        raise RuntimeError(f"only {len(all_dfs)}/{len(SEEDS)} seeds have classification results")

    combined = pd.concat(all_dfs)
    metrics = ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "AUROC", "AUPRC"]
    rows = []
    for (imp, clf), grp in combined.groupby(["ImpMethod", "Classifier"]):
        row = {"ImpMethod": imp, "Classifier": clf}
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi, n = compute_ci(grp[m].values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo, 4) if np.isfinite(lo) else np.nan
            row[f"{m}_ci_hi"] = round(hi, 4) if np.isfinite(hi) else np.nan
        rows.append(row)

    out_path = os.path.join(RESULTS_DIR, "classification_aggregated.csv")
    pd.DataFrame(rows).sort_values("AUROC_mean", ascending=False).to_csv(out_path, index=False)
    print(f"  table3 -> {out_path}")


def aggregate_imputation(seed_dirs):
    all_dfs = []
    for sd in seed_dirs:
        csv_path = os.path.join(sd, "output", "imputation_all_modalities.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if len(all_dfs) < len(SEEDS):
        raise RuntimeError(f"only {len(all_dfs)}/{len(SEEDS)} seeds have imputation results")

    combined = pd.concat(all_dfs)
    group_keys = ["Modality", "Type", "Method", "Feature"]
    rows = []
    for keys, grp in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        metrics = ["Accuracy", "AUROC"] if row["Type"] == "categorical" else ["MAE", "MSE", "RMSE", "MAPE", "R2"]
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi, n = compute_ci(pd.to_numeric(grp[m], errors="coerce").values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo, 4)   if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi, 4)   if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(RESULTS_DIR, "imputation_aggregated.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  table2 -> {out_path}")


def aggregate_demographics(seed_dirs):
    all_dfs = []
    for sd in seed_dirs:
        path = os.path.join(sd, "output", "classification_by_demographics.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if len(all_dfs) < len(SEEDS):
        raise RuntimeError(f"only {len(all_dfs)}/{len(SEEDS)} seeds have demographic results")

    combined = pd.concat(all_dfs)
    group_keys = ["ImpMethod", "Classifier", "Dimension", "Group"]
    metrics = ["Accuracy", "F1_Macro", "AUROC", "AUPRC"]
    rows = []
    for keys, grp in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        row["N_mean"] = int(round(grp["N"].mean()))
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi, n = compute_ci(pd.to_numeric(grp[m], errors="coerce").values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo,   4) if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi,   4) if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(RESULTS_DIR, "classification_by_demographics_aggregated.csv")
    (pd.DataFrame(rows)
       .sort_values(["Dimension", "Group", "ImpMethod", "AUROC_mean"], ascending=[True, True, True, False])
       .to_csv(out_path, index=False))
    print(f"  demographics -> {out_path}")


def aggregate_moo(seed_dirs, source_csv, out_csv, label):
    all_dfs = []
    for sd in seed_dirs:
        path = os.path.join(sd, "output", source_csv)
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if len(all_dfs) < len(SEEDS):
        raise RuntimeError(f"only {len(all_dfs)}/{len(SEEDS)} seeds have {source_csv}")

    combined = pd.concat(all_dfs)
    group_keys = ["Modality", "Type", "Method", "Feature"]
    rows = []
    for keys, grp in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        metrics = ["Accuracy", "AUROC"] if row["Type"] == "categorical" else ["MAE", "MSE", "RMSE", "MAPE", "R2"]
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi, n = compute_ci(pd.to_numeric(grp[m], errors="coerce").values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo, 4)   if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi, 4)   if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(RESULTS_DIR, out_csv)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  {label} -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed PSN pipeline")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Seeds to run (default: all {SEEDS})"
    )
    parser.add_argument(
        "--steps", nargs="+", default=None, choices=ALL_STEPS,
        help=f"Pipeline steps to run per seed (default: all). Choices: {ALL_STEPS}"
    )
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help="Skip aggregation after running seeds"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run seeds even if done markers are present"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    seeds = args.seeds if args.seeds is not None else SEEDS
    invalid = [s for s in seeds if s not in SEEDS]
    if invalid:
        print(f"Warning: seeds {invalid} not in standard SEEDS {SEEDS}, running anyway")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.steps:
        print(f"seeds: {seeds}  steps: {args.steps}\n")
    else:
        print(f"seeds: {seeds}\n")

    seed_dirs = [os.path.join(RESULTS_DIR, f"seed_{s}") for s in SEEDS]  # all seeds for aggregation

    for i, seed in enumerate(seeds, 1):
        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")

        if not args.force and args.steps is None and all(os.path.exists(os.path.join(seed_dir, m)) for m in DONE_MARKERS):
            print(f"seed {seed} ({i}/{len(seeds)}) already done, skipping")
            continue

        print(f"seed {seed} ({i}/{len(seeds)})")
        if run_one_seed(seed, seed_dir, steps_to_run=args.steps):
            print(f"  done\n")
        else:
            print(f"  seed {seed} failed\n")

    if not args.skip_aggregate:
        aggregate_classification(seed_dirs)
        aggregate_imputation(seed_dirs)
        aggregate_demographics(seed_dirs)


if __name__ == "__main__":
    main()