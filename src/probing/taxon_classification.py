import argparse
import os
import pickle
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def label_taxonomy(metadata: pd.DataFrame, n_labels: int = 10) -> Tuple[pd.DataFrame, Dict[str, int]]:
    top_labels = metadata["Family"].value_counts().nlargest(n_labels + 1).index.tolist()
    if "Unclassified" in top_labels:
        top_labels.remove("Unclassified")
    else:
        top_labels = top_labels[:-1]
    label_dict = {label: idx for idx, label in enumerate(top_labels)}
    label_dict["other"] = n_labels
    def assign_label(family: str) -> str:
        if family in top_labels:
            return family
        else:
            return "other"
    labeled_metadata = metadata.copy()
    labeled_metadata["taxon_label"] = labeled_metadata["Family"].apply(assign_label)
    return labeled_metadata, label_dict


def train_clf(X: np.ndarray, y: np.ndarray, seed: int = 42, max_iter: int = 100) -> LogisticRegression:
    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=max_iter, multi_class="ovr")
    clf.fit(X, y)
    return clf


def evaluate_clf(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    weighted_f1 = f1_score(y, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y, y_pred)
    macro_auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    weighted_auc = roc_auc_score(y, y_prob, multi_class="ovr", average="weighted")
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
    parser.add_argument("--dataset-name", type=str, required=True, choices=["131k", "66k"], help="Dataset name.")
    parser.add_argument("--methods", type=str, nargs="+", required=True, help="List of embedding methods to evaluate.")
    parser.add_argument("--layer-names", type=str, nargs="+", required=True, help="List of layer names to evaluate.")
    parser.add_argument("--n-labels", type=int, default=10, help="Number of top taxonomy labels to consider.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output TSV file for results.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory to save trained models.")
    parser.add_argument("--ignore-other", action="store_true", help="If set, ignore 'other' class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations for logistic regression.")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    set_seed(args.seed)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = pd.read_table(args.metadata)
    labeled_metadata, label_dict = label_taxonomy(metadata, n_labels=args.n_labels)
    y = np.array([label_dict[label] for label in labeled_metadata["taxon_label"].values])
    if args.ignore_other:
        labeled_metadata = labeled_metadata[labeled_metadata["taxon_label"] != "other"]
    
    table = []
    for method in args.methods:
        for layer_name in args.layer_names:
            for fold in range(1, 6):
                embedding_path = f"{args.emb_dir}/{args.dataset_name}/{method}/{layer_name}.pt"
                X = torch.load(embedding_path).float().numpy()
                
                train_df = labeled_metadata[labeled_metadata["split"] != fold]
                test_df = labeled_metadata[labeled_metadata["split"] == fold]
                X_train = X[train_df.index.values, :]
                X_test = X[test_df.index.values, :]
                y_train = y[train_df.index.values]
                y_test = y[test_df.index.values]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model_path = f"{args.model_dir}/{method}_{layer_name}_fold{fold}.pkl"
                if os.path.exists(model_path):
                    clf = pickle.load(open(model_path, "rb"))
                else:
                    clf = train_clf(X_train, y_train, seed=args.seed, max_iter=args.max_iter)
                train_results = evaluate_clf(clf, X_train, y_train)
                test_results = evaluate_clf(clf, X_test, y_test)
                result = {
                    "method": method,
                    "layer_name": layer_name,
                    "fold": fold,
                    "train_macro_f1": train_results["macro_f1"],
                    "train_weighted_f1": train_results["weighted_f1"],
                    "train_balanced_accuracy": train_results["balanced_accuracy"],
                    "train_macro_auc": train_results["macro_auc"],
                    "train_weighted_auc": train_results["weighted_auc"],
                    "test_macro_f1": test_results["macro_f1"],
                    "test_weighted_f1": test_results["weighted_f1"],
                    "test_balanced_accuracy": test_results["balanced_accuracy"],
                    "test_macro_auc": test_results["macro_auc"],
                    "test_weighted_auc": test_results["weighted_auc"]
                }
                table.append(result)
                
                pickle.dump(clf, open(model_path, "wb"))
    
    df_output = pd.DataFrame(table)
    df_output.to_csv(args.output_file, sep="\t", index=False)
    print("Unique classes in this experiment:", labeled_metadata["taxon_label"].unique().tolist())
