import argparse
import os
import pickle
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_clf(X: np.ndarray, y: np.ndarray, seed: int = 42, max_iter: int = 100) -> LogisticRegression:
    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=max_iter)
    clf.fit(X, y)
    return clf


def evaluate_clf(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    f1 = f1_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    return {"f1": f1, "balanced_accuracy": bal_acc, "mcc": mcc, "roc_auc": roc_auc}


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob[:, 1] > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    return {"f1": f1, "balanced_accuracy": bal_acc, "mcc": mcc, "roc_auc": roc_auc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, required=True, help="Directory containing embeddings.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing labels.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset")
    parser.add_argument("--layer-names", type=str, nargs="+", required=True, help="List of layer names to evaluate.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output TSV file for results.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory to save trained models.")
    parser.add_argument("--raw-output", type=Path, default=None, help="Optional directory to save raw predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations for logistic regression.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output file.")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    set_seed(args.seed)
    
    metadata = pd.read_table(args.dataset)
    
    table = []
    for layer_name in args.layer_names:
        for i in range(1, 6):
            train_accessions = metadata[metadata["fold"] != i]["accession"].values
            test_accessions = metadata[metadata["fold"] == i]["accession"].values
            
            train_embs = [torch.load(f"{args.emb_dir}/{layer_name}/{accession}.pt").float() for accession in train_accessions]
            test_embs = [torch.load(f"{args.emb_dir}/{layer_name}/{accession}.pt").float() for accession in test_accessions]
            X_train = torch.cat(train_embs, dim=0).numpy()
            X_test = torch.cat(test_embs, dim=0).numpy()
            print(X_train.shape, X_test.shape)
            
            train_labels = [np.load(f"{args.label_dir}/{accession}_cds_annotation.npy") for accession in train_accessions]
            test_labels = [np.load(f"{args.label_dir}/{accession}_cds_annotation.npy") for accession in test_accessions]
            y_train = np.concatenate(train_labels)
            y_test = np.concatenate(test_labels)
            print(y_train.shape, y_test.shape)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if not args.overwrite:
                trained_clf_path = args.model_dir / layer_name / f"fold{i}_trained_model.pkl"
                if trained_clf_path.exists():
                    print(f"Model for layer {layer_name} fold {i} already exists. Skipping training.")
                    clf = pickle.load(open(trained_clf_path, "rb"))
                    train_metrics = evaluate_clf(clf, X_train, y_train)
                    test_metrics = evaluate_clf(clf, X_test, y_test)
                    result = {
                        "layer_name": layer_name,
                        "fold": i,
                        "train_f1_score": train_metrics["f1"],
                        "train_accuracy": train_metrics["balanced_accuracy"],
                        "train_mcc": train_metrics["mcc"],
                        "train_roc_auc": train_metrics["roc_auc"],
                        "test_f1_score": test_metrics["f1"],
                        "test_accuracy": test_metrics["balanced_accuracy"],
                        "test_mcc": test_metrics["mcc"],
                        "test_roc_auc": test_metrics["roc_auc"]
                    }
                    table.append(result)
                    continue
                else:
                    raise FileNotFoundError(f"Trained model for layer {layer_name} fold {i} not found. Please run with --overwrite to train new models.")
            
            clf = train_clf(X_train, y_train, seed=args.seed, max_iter=args.max_iter)
            train_metrics = evaluate_clf(clf, X_train, y_train)
            test_metrics = evaluate_clf(clf, X_test, y_test)
            result = {
                "layer_name": layer_name,
                "fold": i,
                "train_f1_score": train_metrics["f1"],
                "train_accuracy": train_metrics["balanced_accuracy"],
                "train_mcc": train_metrics["mcc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "test_f1_score": test_metrics["f1"],
                "test_accuracy": test_metrics["balanced_accuracy"],
                "test_mcc": test_metrics["mcc"],
                "test_roc_auc": test_metrics["roc_auc"]
            }
            table.append(result)
            
            model_dir = args.model_dir / layer_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"fold{i}_trained_model.pkl"
            pickle.dump(clf, open(model_path, "wb"))
            
            if args.raw_output is not None:
                raw_output_dir = args.raw_output / layer_name
                raw_output_dir.mkdir(parents=True, exist_ok=True)
                np.save(raw_output_dir / f"train_predictions_fold{i}.npy", clf.predict_proba(X_train))
                np.save(raw_output_dir / f"test_predictions_fold{i}.npy", clf.predict_proba(X_test))
    
    df_output = pd.DataFrame(table)
    df_output.to_csv(args.output_file, sep="\t", index=False)
