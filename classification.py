import argparse
import csv
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

from data_loader import get_or_create_split
from evaluation import plot_classification_results, save_classification_csv
from utils import set_deterministic_seed
import config


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


def save_predictions_csv(results_store, patient_ids, path):
    """Per-patient predictions for every (ImpMethod, Classifier) combination."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["ImpMethod", "Classifier", "patient_id", "ground_truth_label", "predict_label", "predict_prob_0", "predict_prob_1", "predict_prob_2"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, result in results_store.items():
            imp_method, clf_name = key.rsplit(" - ", 1)
            y_true = result["y_true"]
            y_pred = result["y_pred"]
            y_prob = result["y_prob"]
            for i, pid in enumerate(patient_ids):
                writer.writerow({
                    "ImpMethod": imp_method, "Classifier": clf_name, "patient_id": pid,
                    "ground_truth_label": int(y_true[i]), "predict_label": int(y_pred[i]),
                    "predict_prob_0": round(float(y_prob[i, 0]), 6),
                    "predict_prob_1": round(float(y_prob[i, 1]), 6),
                    "predict_prob_2": round(float(y_prob[i, 2]), 6),
                })
    print(f"Saved predictions: {path}")


def evaluate_classifiers(X_train, y_train, X_test, y_test, label, results_store, scale=True, seed=42):
    """Evaluate classification on features or embeddings"""

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    set_deterministic_seed(seed)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', class_weight="balanced", random_state=config.config["random_state"]),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=config.config["random_state"]),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
            random_state=config.config["random_state"], objective='multi:softprob', num_class=3, eval_metric='mlogloss'
        )
    }

    temp_results = {}
    for clf_name, clf in classifiers.items():
        set_deterministic_seed(seed)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "auroc_ovr": roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'),
            "auprc": average_precision_score(y_test_bin, y_prob, average='macro'),
        }
        temp_results[clf_name] = {"metrics": metrics, "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}

    # Sort by AUROC (descending)
    sorted_clfs = sorted(temp_results.items(), key=lambda x: x[1]["metrics"]["auroc_ovr"], reverse=True)

    print(f"\n{label} Classification Results:")
    print("=" * 80)
    for clf_name, result in sorted_clfs:
        metrics = result["metrics"]
        key = f"{label} - {clf_name}"
        results_store[key] = result
        print(f"{clf_name}: Acc={metrics['accuracy']:.3f}, F1_macro={metrics['f1_macro']:.3f}, AUROC_ovr={metrics['auroc_ovr']:.3f}")


def main():
    parse_and_override(config.config)
    seed = config.config["random_state"]
    set_deterministic_seed(seed)
    OUT = config.config["output_dir"]
    out_dir = os.path.join(OUT, "output")
    imp_dir = os.path.join(OUT, "imputations")
    emb_dir = os.path.join(OUT, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Load train and test data
    train_data, test_data, _ = get_or_create_split(config.config, save=True)
    y_train = train_data["labels"]
    y_test = test_data["labels"]

    # Modalities to impute
    modalities_to_impute = config.config["modalities_to_impute"]
    print(f"\n[INFO] Imputed modalities: {modalities_to_impute}")

    # ========== EMBEDDING-BASED CLASSIFICATION ==========
    print("\n" + "=" * 80)
    print("EMBEDDING-BASED CLASSIFICATION (Graph-Learned Representations)")
    print("=" * 80)

    embedding_train_path = os.path.join(emb_dir, "train_embeddings.npy")
    embedding_test_path = os.path.join(emb_dir, "test_embeddings.npy")

    classification_results = {}

    if os.path.exists(embedding_train_path) and os.path.exists(embedding_test_path):
        train_embeddings = np.load(embedding_train_path)
        test_embeddings = np.load(embedding_test_path)
        print(f"Train embeddings shape: {train_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")
        evaluate_classifiers(train_embeddings, y_train, test_embeddings, y_test, label="Graph-Embeddings", results_store=classification_results, scale=False, seed=seed)

    # ========== FEATURE-BASED CLASSIFICATION ==========
    print("\n" + "=" * 80)
    print("FEATURE ORIGINAL SCALE - BASED CLASSIFICATION (Imputed Features)")
    print("=" * 80)

    imputations_dir = imp_dir
    imputation_methods = ["model", "mean", "median", "knn1", "knn3", "knn", "linearregression", "gain", "grape"]
    missing_files = []
    for method in imputation_methods:
        for split in ["train", "test"]:
            for mod in modalities_to_impute:
                filepath = os.path.join(imputations_dir, f"{split}_{method}_{mod}.npy")
                if not os.path.exists(filepath): missing_files.append(filepath)

    if missing_files:
        print(f"  Some imputation files are missing ({len(missing_files)} total), will skip those methods:")
        for f in missing_files:
            print(f"    {f}")

    missing_methods = {os.path.basename(f).split('_')[1] for f in missing_files}

    # load imputed arrays for each method; skip any method that had missing files above
    imputation_data = {}
    method_name_map = {
        "knn1": "KNN1",
        "knn3": "KNN3",
        "knn": "KNN",
        "linearregression": "LinearRegression",
        "gain": "GAIN",
        "grape": "GRAPE",
    }
    for method in imputation_methods:
        if method in missing_methods:
            print(f"  skipping '{method}' (files not found)")
            continue

        method_name = method_name_map.get(method, method.capitalize())
        imputation_data[method_name] = {"train": {}, "test": {}}

        for split in ["train", "test"]:
            for mod in modalities_to_impute:
                filepath = os.path.join(imputations_dir, f"{split}_{method}_{mod}.npy")
                imputation_data[method_name][split][mod] = np.load(filepath)
        print(f"  Loaded: {method_name}")

    for method_name, data in imputation_data.items():
        # Build feature matrix: M1, M2, [M3, M5, M6, M7 - imputed], M4
        train_features = [train_data["m1"], train_data["m2"]]
        test_features = [test_data["m1"], test_data["m2"]]

        # Imputed modalities (M3, M5, M6, M7)
        for mod in modalities_to_impute:
            train_features.append(data["train"][mod])
            test_features.append(data["test"][mod])

        # M4 modality
        train_features.append(train_data["m4"])
        test_features.append(test_data["m4"])

        # Concatenate all features
        X_train_imp = np.concatenate(train_features, axis=1)
        X_test_imp = np.concatenate(test_features, axis=1)

        # Evaluate
        evaluate_classifiers(X_train_imp, y_train, X_test_imp, y_test, label=method_name, results_store=classification_results, scale=True, seed=seed)

    if not classification_results:
        print("\nNo classification results to save (no methods succeeded).")
        return

    plot_dir = os.path.join(OUT, "plots")
    plot_classification_results(classification_results, save_dir=plot_dir)

    save_classification_csv(classification_results, os.path.join(out_dir, "classification_all_methods.csv"))
    save_predictions_csv(classification_results, test_data["patient_ids"], os.path.join(out_dir, "predictions_all_methods.csv"))


if __name__ == "__main__":
    main()
