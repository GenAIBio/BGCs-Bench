import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_clf(X: np.ndarray, y: np.ndarray, seed: int = 42, max_iter: int = 100) -> OneVsRestClassifier:
    base_clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=max_iter)
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X, y)
    return clf


def multilabel_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for j in range(y_true.shape[1]):
        unique_vals = np.unique(y_true[:, j])
        if len(unique_vals) < 2:
            continue
        score = balanced_accuracy_score(y_true[:, j], y_pred[:, j])
        scores.append(score)
    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))


def multilabel_roc_auc(y_true: np.ndarray, y_score: np.ndarray, average: str = "weighted") -> float:
    aucs = []
    weights = []
    for j in range(y_true.shape[1]):
        y_true_j = y_true[:, j]
        y_score_j = y_score[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        auc_j = roc_auc_score(y_true_j, y_score_j)
        aucs.append(auc_j)
        weights.append(np.sum(y_true_j))
    if len(aucs) == 0:
        return np.nan
    aucs = np.array(aucs, dtype=float)
    weights = np.array(weights, dtype=float)
    if average == "macro":
        return float(np.mean(aucs))
    elif average == "weighted":
        if np.sum(weights) == 0:
            return float(np.mean(aucs))
        return float(np.average(aucs, weights=weights))
    else:
        raise ValueError(f"Unsupported average: {average}. Use 'macro' or 'weighted'.")


def evaluate_clf(clf: OneVsRestClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    weighted_f1 = f1_score(y, y_pred, average="weighted")
    bal_acc = multilabel_balanced_accuracy(y, y_pred)
    macro_auc = multilabel_roc_auc(y, y_prob, average="macro")
    weighted_auc = multilabel_roc_auc(y, y_prob, average="weighted")
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, required=True, help="Directory containing embeddings.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata TSV file.")
    parser.add_argument("--dataset-name", type=str, required=True, choices=["full", "half"], help="Dataset name.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output TSV file for results.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory to save trained models.")
    parser.add_argument("--ignore-other", action="store_true", help="If set, ignore 'other' class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations for logistic regression.")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    set_seed(args.seed)
    
    CLASSES = ["NRPS", "PKS", "ribosomal", "terpene", "saccharide", "other"]
    
    metadata = pd.read_table(args.metadata)
    y = metadata[CLASSES].values
    if args.ignore_other:
        metadata = metadata[metadata["class"] != "other"]
        y = y[:, :-1]
    
    table = []
    for fold in range(1, 6):
        for k in range(3, 10):
            X = np.load(f"{args.emb_dir}/{args.dataset_name}/kmer_profile_k{k}.npy")
            
            train_df = metadata[metadata["split"] != fold]
            test_df = metadata[metadata["split"] == fold]
            X_train = X[train_df.index.values, :]
            X_test = X[test_df.index.values, :]
            y_train = y[train_df.index.values]
            y_test = y[test_df.index.values]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = train_clf(X_train, y_train, seed=args.seed, max_iter=args.max_iter)
            train_metrics = evaluate_clf(clf, X_train, y_train)
            test_metrics = evaluate_clf(clf, X_test, y_test)
            result = {
                "k-mer": k,
                "fold": fold,
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_macro_auc": train_metrics["macro_auc"],
                "train_weighted_auc": train_metrics["weighted_auc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_auc": test_metrics["macro_auc"],
                "test_weighted_auc": test_metrics["weighted_auc"]
            }
            table.append(result)
            
            model_path = f"{args.model_dir}/{k}mer_fold{fold}.pkl"
            pickle.dump(clf, open(model_path, "wb"))
    
    df_output = pd.DataFrame(table)
    df_output.to_csv(args.output_file, sep="\t", index=False)
    print("Unique classes in this experiment:", np.unique(y_test))
