import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [1, 2, 3, 4, 42]
RESULTS_DIR = "results_ablation"

'''
STAGES = [
    ("S1_encoder_decoder",   ["--skip-graph", "1", "--lambda-cl", "0.0"]),
    ("S2_average_fusion",    ["--skip-graph", "0", "--filter-graph", "0",
                               "--fusion-mode", "average", "--lambda-cl", "0.0"]),
    ("S3_gated_fusion",      ["--skip-graph", "0", "--filter-graph", "0",
                               "--fusion-mode", "gated",   "--lambda-cl", "0.0"]),
    ("S4_topk_filter",       ["--skip-graph", "0", "--filter-graph", "1",
                               "--fusion-mode", "gated",   "--lambda-cl", "0.0"]),
    ("S5_no_reliable_ratio", ["--skip-graph", "0", "--filter-graph", "1",
                               "--fusion-mode", "gated",   "--lambda-cl", "1.0",
                               "--reliable-ratio", "0"]),
    ("S6_no_topk",           ["--skip-graph", "0", "--filter-graph", "0",
                               "--fusion-mode", "gated",   "--lambda-cl", "1.0"]),
    ("S7_no_gated_fusion",   ["--skip-graph", "0", "--filter-graph", "1",
                               "--fusion-mode", "average", "--lambda-cl", "1.0"]),
]
'''

STAGES = [
    # ── modality_a=1 (absent-modality patients masked from GCN neighbors) ─────
    ("S_trim_topk1_rr1_ma1",    ["--skip-graph", "0", "--filter-graph", "1", "--top-k", "1","--reliable-ratio", "1", "--modality-a", "1"]),
    ("S_trim_topk3_rr1_ma1",    ["--skip-graph", "0", "--filter-graph", "1", "--top-k", "3","--reliable-ratio", "1", "--modality-a", "1"]),
    ("S_trim_topk5_rr1_ma1",    ["--skip-graph", "0", "--filter-graph", "1", "--top-k", "5","--reliable-ratio", "1", "--modality-a", "1"]),
    ("S_trim_topk10_rr1_ma1",   ["--skip-graph", "0", "--filter-graph", "1", "--top-k", "10","--reliable-ratio", "1", "--modality-a", "1"]),
    ("S_trim_topk20_rr1_ma1",   ["--skip-graph", "0", "--filter-graph", "1", "--top-k", "20","--reliable-ratio", "1", "--modality-a", "1"]),
    ("S_trim_nofilter_rr1_ma1", ["--skip-graph", "0", "--filter-graph", "0", "--reliable-ratio", "1", "--modality-a", "1"]),
]

DONE_MARKERS = [
    "output/classification_all_methods.csv",
    "output/imputation_all_modalities.csv",
    "output/best_model.pth",
    "output/best_model_finetuned.pth",
    "output/classification_by_demographics.csv",
    "imputations/test_model_m3.npy",
    "imputations/test_model_m5.npy",
    "imputations/test_model_m6.npy",
    "imputations/test_model_m7.npy",
    "embeddings/train_embeddings.npy",
    "embeddings/test_embeddings.npy",
]


def run_command(command):
    process = subprocess.Popen(command, shell=True, universal_newlines=True)
    process.wait()
    if process.returncode != 0:
        print(f"  {command.split()[1]} failed (exit {process.returncode})")
        return False
    return True


def args_to_str(extra_args):
    return " ".join(extra_args)


ALL_STEPS = ["train", "decoder_mod", "imputation", "error_analysis", "classification", "eval_demographics"]


def run_one_seed_stage(seed, stage_name, extra_args, stage_seed_dir, steps_to_run=None):
    """Run the full pipeline for one seed + one stage."""
    os.makedirs(stage_seed_dir, exist_ok=True)
    out_dir_arg = f"--output-dir {stage_seed_dir}"
    seed_arg    = f"--random-state {seed}"
    extra       = args_to_str(extra_args)
    base        = f"{sys.executable}"

    all_steps = [
        (f"{base} train.py {out_dir_arg} {seed_arg} {extra}",            True),
        (f"{base} decoder_mod.py {out_dir_arg} {seed_arg} {extra}",      True),
        (f"{base} imputation.py {out_dir_arg} {seed_arg} {extra}",       True),
        #(f"{base} error_analysis.py {out_dir_arg} {seed_arg}",           False),
        (f"{base} classification.py {out_dir_arg} {seed_arg}",           True),
        #(f"{base} eval_demographics.py {out_dir_arg} {seed_arg}",        False),
    ]

    steps = [(cmd, crit) for (cmd, crit), name in zip(all_steps, ALL_STEPS)
             if steps_to_run is None or name in steps_to_run]

    for cmd, critical in steps:
        script = cmd.split()[1]
        print(f"    {script}")
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


def aggregate_classification(stage_name, seed_dirs, out_base):
    all_dfs = []
    for sd in seed_dirs:
        path = os.path.join(sd, "output", "classification_all_methods.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if not all_dfs:
        print(f"  [{stage_name}] no classification results to aggregate")
        return

    combined = pd.concat(all_dfs)
    metrics = ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "AUROC", "AUPRC"]
    rows = []
    for (imp, clf), grp in combined.groupby(["ImpMethod", "Classifier"]):
        row = {"ImpMethod": imp, "Classifier": clf}
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi, _ = compute_ci(grp[m].values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo,   4) if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi,   4) if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(out_base, "classification_aggregated.csv")
    pd.DataFrame(rows).sort_values("AUROC_mean", ascending=False).to_csv(out_path, index=False)
    print(f"  classification -> {out_path}")


def aggregate_imputation(stage_name, seed_dirs, out_base):
    all_dfs = []
    for sd in seed_dirs:
        path = os.path.join(sd, "output", "imputation_all_modalities.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if not all_dfs:
        print(f"  [{stage_name}] no imputation results to aggregate")
        return

    combined = pd.concat(all_dfs)
    group_keys = ["Modality", "Type", "Method", "Feature"]
    rows = []
    for keys, grp in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        mets = ["Accuracy", "AUROC"] if row["Type"] == "categorical" else ["MAE", "MSE", "RMSE", "MAPE", "R2"]
        for m in mets:
            if m not in grp.columns:
                continue
            mean, lo, hi, _ = compute_ci(pd.to_numeric(grp[m], errors="coerce").values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo,   4) if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi,   4) if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(out_base, "imputation_aggregated.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  imputation -> {out_path}")


def aggregate_demographics(stage_name, seed_dirs, out_base):
    all_dfs = []
    for sd in seed_dirs:
        path = os.path.join(sd, "output", "classification_by_demographics.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if not all_dfs:
        return

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
            mean, lo, hi, _ = compute_ci(pd.to_numeric(grp[m], errors="coerce").values)
            row[f"{m}_mean"]  = round(mean, 4) if np.isfinite(mean) else np.nan
            row[f"{m}_ci_lo"] = round(lo,   4) if np.isfinite(lo)   else np.nan
            row[f"{m}_ci_hi"] = round(hi,   4) if np.isfinite(hi)   else np.nan
        rows.append(row)

    out_path = os.path.join(out_base, "demographics_aggregated.csv")
    (pd.DataFrame(rows)
       .sort_values(["Dimension", "Group", "ImpMethod", "AUROC_mean"],
                    ascending=[True, True, True, False])
       .to_csv(out_path, index=False))
    print(f"  demographics -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation study across seeds and stages")
    p.add_argument(
        "--stage", type=str, default=None,
        help="Stage name (e.g. S1_encoder_decoder) or 1-based index (e.g. 1). "
             "Omit to run all stages."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Seeds to run (default: all {SEEDS})"
    )
    p.add_argument(
        "--steps", nargs="+", default=None, choices=ALL_STEPS,
        help=f"Pipeline steps to run per seed (default: all). Choices: {ALL_STEPS}"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run seeds even if done markers are present"
    )
    p.add_argument(
        "--skip-aggregate", action="store_true",
        help="Skip aggregation after running seeds"
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stage_names = [s for s, _ in STAGES]
    seeds = args.seeds if args.seeds is not None else SEEDS

    # Filter to requested stage if provided
    if args.stage is not None:
        try:
            idx = int(args.stage) - 1
            if not (0 <= idx < len(STAGES)):
                raise ValueError(f"Stage index {idx + 1} out of range (1–{len(STAGES)})")
            selected = [STAGES[idx]]
        except ValueError as e:
            if args.stage in stage_names:
                selected = [STAGES[stage_names.index(args.stage)]]
            else:
                raise SystemExit(
                    f"Unknown stage '{args.stage}'. "
                    f"Valid names: {stage_names}\n"
                    f"Or pass a 1-based index 1–{len(STAGES)}."
                ) from e
    else:
        selected = STAGES

    print(f"seeds:  {seeds}")
    if args.steps:
        print(f"steps:  {args.steps}")
    print(f"stages: {[s for s, _ in selected]}\n")

    for stage_name, extra_args in selected:
        stage_dir = os.path.join(RESULTS_DIR, stage_name)
        os.makedirs(stage_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name}")
        print(f"args:  {args_to_str(extra_args)}")
        print(f"{'='*60}")

        seed_dirs = [os.path.join(stage_dir, f"seed_{s}") for s in SEEDS]  # all seeds for aggregation

        for i, seed in enumerate(seeds, 1):
            seed_dir = os.path.join(stage_dir, f"seed_{seed}")

            if not args.force and args.steps is None and all(os.path.exists(os.path.join(seed_dir, m)) for m in DONE_MARKERS):
                print(f"  seed {seed} ({i}/{len(seeds)}) already done, skipping")
                continue

            print(f"  seed {seed} ({i}/{len(seeds)})")
            if run_one_seed_stage(seed, stage_name, extra_args, seed_dir, steps_to_run=args.steps):
                print(f"    done")
            else:
                print(f"    seed {seed} FAILED")

        if not args.skip_aggregate:
            print(f"\n  Aggregating {stage_name} ...")
            aggregate_classification(stage_name, seed_dirs, stage_dir)
            aggregate_imputation(stage_name, seed_dirs, stage_dir)
            aggregate_demographics(stage_name, seed_dirs, stage_dir)

    print(f"\nAll stages complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
