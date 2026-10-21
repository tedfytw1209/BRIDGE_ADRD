"""
Trains classifiers on the full training set (same setup as classification.py), then evaluates AUROC/AUPRC/Accuracy/F1_Macro
on demographic subsets of the test set.

Demographic dimensions:
  Sex  : Female (SEX=0), Male (SEX=1)
  Age  : age<50, age_50-65, age_65-75, age_75+
  Race : White (RACE=1.0), Black (RACE=2.0)
         [Other race groups have < 10 test patients and are skipped]

Groups with fewer than MIN_GROUP_N patients, or with only one class present, are skipped.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

import argparse

import config
from data_loader import get_or_create_split
from utils import set_deterministic_seed


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
    p.add_argument('--wandb-run-name',  type=str,   default=None)
    args, _ = p.parse_known_args()

    if args.output_dir     is not None: cfg['output_dir']     = args.output_dir
    if args.random_state   is not None: cfg['random_state']   = args.random_state
    if args.skip_graph     is not None: cfg['skip_graph']     = args.skip_graph
    if args.filter_graph   is not None: cfg['filter_graph']   = args.filter_graph
    if args.fusion_mode    is not None: cfg['fusion_mode']    = args.fusion_mode
    if args.lambda_cl      is not None: cfg['lambda_cl']      = args.lambda_cl
    if args.reliable_ratio is not None: cfg['reliable_ratio'] = args.reliable_ratio

    out = cfg['output_dir']
    cfg['save_model_path']     = os.path.join(out, 'output', 'best_model.pth')
    cfg['finetune_model_path'] = os.path.join(out, 'output', 'best_model_finetuned.pth')

    return args

MIN_GROUP_N = 10  # skip group if fewer than
CSV_FIELDS = ["ImpMethod", "Classifier", "Dimension", "Group", "N", "Accuracy", "F1_Macro", "AUROC", "AUPRC",]


def load_demo_df(path, test_ids):
    """Load m1_features.csv and align rows to test patient IDs."""
    df = pd.read_csv(path).set_index("subid")
    return df.reindex(test_ids)


def get_demo_groups(demo_df):
    """Returns list of (dimension_label, group_label, bool_mask) for test patients."""
    groups = []

    # Sex
    for val, label in [(0, "Female"), (1, "Male")]:
        mask = (demo_df["SEX"] == val).values
        groups.append(("Sex", label, mask))

    # Age bins (as per m1_features.csv)
    for col, label in [("age<50", "Age<50"), ("age_50-65", "Age_50-65"), ("age_65-75", "Age_65-75"), ("age_75+", "Age_75+")]:
        if col in demo_df.columns:
            mask = (demo_df[col] == 1).values
            groups.append(("Age", label, mask))

    # Race - only White and Black have sufficient n in test set
    for val, label in [(1.0, "White"), (2.0, "Black")]:
        mask = (demo_df["RACE"] == val).values
        groups.append(("Race", label, mask))

    return groups


def _fmt(v):
    return round(float(v), 4) if np.isfinite(v) else "N/A"


def eval_subgroup(y_true, y_pred, y_prob, mask):
    """
    Compute Accuracy, F1_Macro, AUROC, AUPRC on the boolean-masked subgroup.
    Returns None if the group is too small or has only one class present.
    """
    yt  = y_true[mask]
    yp  = y_pred[mask]
    ypr = y_prob[mask]
    n   = int(mask.sum())

    if n < MIN_GROUP_N:
        return None
    if len(np.unique(yt)) < 2:
        return None

    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, average="macro", zero_division=0)

    auroc = np.nan
    auprc = np.nan
    try:
        yt_bin = label_binarize(yt, classes=[0, 1, 2])
        auroc  = roc_auc_score(yt_bin, ypr, multi_class="ovr", average="macro")
        auprc  = average_precision_score(yt_bin, ypr, average="macro")
    except Exception:
        pass  # subgroup may lack all 3 classes - AUROC/AUPRC reported as N/A

    return {"N": n, "Accuracy": _fmt(acc), "F1_Macro": _fmt(f1), "AUROC": _fmt(auroc), "AUPRC": _fmt(auprc)}


def _make_classifiers(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed,),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed,),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
            random_state=seed, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        ),
    }


def main():
    parse_and_override(config.config)
    seed = config.config["random_state"]
    set_deterministic_seed(seed)
    OUT = config.config["output_dir"]
    out_dir = os.path.join(OUT, "output")
    imp_dir = os.path.join(OUT, "imputations")
    emb_dir = os.path.join(OUT, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    train_data, test_data, _ = get_or_create_split(config.config, save=True)
    y_train  = train_data["labels"]
    y_test   = test_data["labels"]
    test_ids = test_data["patient_ids"]

    # demographic lookup table aligned to test patient order
    demo_path = config.config["data_paths"]["m1_features"]
    demo_df   = load_demo_df(demo_path, test_ids)
    groups    = get_demo_groups(demo_df)

    modalities_to_impute = config.config["modalities_to_impute"]
    imputations_dir = imp_dir
    imputation_methods = [
        "model", "mean", "median", "knn", "linearregression", "mice", "missforest", "ppm", "gain", "grape",
    ]
    method_name_map = {"knn": "KNN", "linearregression": "LinearRegression", "mice": "MICE", "missforest": "MissForest", "ppm": "PPM", "gain": "GAIN", "grape": "GRAPE",}

    rows = []

    def run_method(X_tr, X_te, method_name):
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        print(f"  {method_name}")
        for clf_name, clf in _make_classifiers(seed).items():
            clf.fit(X_tr_sc, y_train)
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)

            for dim, grp_label, mask in groups:
                result = eval_subgroup(y_test, y_pred, y_prob, mask)
                if result is None:
                    continue
                rows.append({"ImpMethod": method_name, "Classifier": clf_name, "Dimension": dim, "Group": grp_label, **result})

    # ── Graph embeddings ──
    emb_tr = os.path.join(emb_dir, "train_embeddings.npy")
    emb_te = os.path.join(emb_dir, "test_embeddings.npy")
    if os.path.exists(emb_tr) and os.path.exists(emb_te):
        run_method(np.load(emb_tr), np.load(emb_te), "Graph-Embeddings")

    # ── Imputation-based feature vectors ──
    for method in imputation_methods:
        files_ok = all(
            os.path.exists(os.path.join(imputations_dir, f"{split}_{method}_{mod}.npy"))
            for split in ["train", "test"]
            for mod in modalities_to_impute
        )
        if not files_ok:
            continue

        method_name = method_name_map.get(method, method.capitalize())

        tr_feats = [train_data["m1"], train_data["m2"]]
        te_feats = [test_data["m1"],  test_data["m2"]]
        for mod in modalities_to_impute:
            tr_feats.append(np.load(os.path.join(imputations_dir, f"train_{method}_{mod}.npy")))
            te_feats.append(np.load(os.path.join(imputations_dir, f"test_{method}_{mod}.npy")))
        tr_feats.append(train_data["m4"])
        te_feats.append(test_data["m4"])

        run_method(np.concatenate(tr_feats, axis=1), np.concatenate(te_feats, axis=1), method_name)

    out_path = os.path.join(out_dir, "classification_by_demographics.csv")
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
