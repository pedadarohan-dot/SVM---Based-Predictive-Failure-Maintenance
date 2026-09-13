"""
train.py — SVM-based predictive maintenance on the AI4I 2020 dataset.

Trains and compares Linear, Polynomial, and RBF-kernel SVMs on five
continuous machine-operating parameters to predict binary machine failure,
then tunes the RBF model with grid search and reports the precision/recall
trade-off that tuning produces.

Usage:
    python train.py --data ai4i2020.csv
    python train.py --data ai4i2020.csv --no-plots     # metrics only, no figures
    python train.py --data ai4i2020.csv --save-plots outputs/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
RANDOM_STATE = 42


@dataclass
class Metrics:
    """Accuracy / precision / recall / F1 for one model's test-set predictions."""

    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_predictions(cls, model: str, y_true, y_pred) -> "Metrics":
        return cls(
            model=model,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path} — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def describe_data(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    print("\nClass distribution:")
    print(f"  Normal (0):  {counts[0]:>6d}  ({pct[0]:.2f}%)")
    print(f"  Failure (1): {counts[1]:>6d}  ({pct[1]:.2f}%)")
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")


def make_splits(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain/test split: {len(X_train)} / {len(X_test)} (stratified)")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #

def train_kernel(kernel: str, X_train, y_train, **svc_kwargs) -> SVC:
    model = SVC(kernel=kernel, class_weight="balanced", **svc_kwargs)
    model.fit(X_train, y_train)
    return model


def report(name: str, y_test, y_pred) -> Metrics:
    m = Metrics.from_predictions(name, y_test, y_pred)
    print(f"\n{name}")
    print(f"  Accuracy={m.accuracy:.4f}  Precision={m.precision:.4f}  "
          f"Recall={m.recall:.4f}  F1={m.f1:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    return m


def tune_rbf(X_train, y_train) -> GridSearchCV:
    """Grid search (C, gamma) for the RBF kernel, scored on F1, 5-fold CV."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced")),
    ])
    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    }
    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"\nBest params: {grid.best_params_}  (CV F1={grid.best_score_:.4f})")
    return grid


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _finish(fig, save_path: Path | None, name: str):
    fig.tight_layout()
    if save_path:
        out = save_path / f"{name}.png"
        fig.savefig(out, dpi=150)
        print(f"  saved {out}")
        plt.close(fig)
    else:
        plt.show()


def plot_class_distribution(df: pd.DataFrame, save_path: Path | None):
    counts = df[TARGET].value_counts().sort_index()
    fig = plt.figure(figsize=(7, 5))
    bars = plt.bar(["Normal Operation", "Machine Failure"], counts.values)
    plt.title("Machine Failure Class Distribution")
    plt.ylabel("Number of Samples")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                  ha="center", va="bottom")
    _finish(fig, save_path, "class_distribution")


def plot_model_comparison(results: list[Metrics], save_path: Path | None):
    labels = [r.model for r in results]
    series = {
        "Accuracy": [r.accuracy for r in results],
        "Precision": [r.precision for r in results],
        "Recall": [r.recall for r in results],
        "F1 Score": [r.f1 for r in results],
    }
    x = np.arange(len(labels))
    width = 0.2
    fig = plt.figure(figsize=(11, 6))
    for i, (name, values) in enumerate(series.items()):
        plt.bar(x + (i - 1.5) * width, values, width, label=name)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Comparison of SVM Models")
    plt.legend()
    _finish(fig, save_path, "model_comparison")


def plot_confusion(y_test, y_pred, title: str, save_path: Path | None):
    fig = plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Normal", "Failure"])
    plt.title(title)
    _finish(fig, save_path, "confusion_matrix_optimized")


def plot_error_comparison(rbf_pred, best_pred, y_test, save_path: Path | None):
    rbf_cm, best_cm = confusion_matrix(y_test, rbf_pred), confusion_matrix(y_test, best_pred)
    labels = ["RBF SVM", "Optimized RBF"]
    fp = [rbf_cm[0, 1], best_cm[0, 1]]
    fn = [rbf_cm[1, 0], best_cm[1, 0]]
    x = np.arange(len(labels))
    width = 0.35
    fig = plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, fp, width, label="False Positives")
    plt.bar(x + width / 2, fn, width, label="False Negatives")
    plt.xticks(x, labels)
    plt.ylabel("Number of Samples")
    plt.title("Effect of Hyperparameter Tuning on Prediction Errors")
    plt.legend()
    _finish(fig, save_path, "error_comparison")


def plot_precision_recall(results: list[Metrics], save_path: Path | None):
    fig = plt.figure(figsize=(8, 6))
    for r in results:
        plt.scatter(r.precision, r.recall, s=100)
        plt.annotate(r.model, (r.precision, r.recall), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Comparison of SVM Models")
    plt.grid(True, alpha=0.3)
    _finish(fig, save_path, "precision_recall")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ai4i2020.csv", help="path to the AI4I 2020 CSV")
    parser.add_argument("--no-plots", action="store_true", help="skip all plotting")
    parser.add_argument("--save-plots", metavar="DIR", default=None,
                         help="save figures to DIR instead of displaying them")
    args = parser.parse_args()

    save_path = None
    if args.save_plots:
        save_path = Path(args.save_plots)
        save_path.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    describe_data(df)
    X_train, X_test, y_train, y_test = make_splits(df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results: list[Metrics] = []
    predictions: dict[str, np.ndarray] = {}

    for name, kernel, kwargs in [
        ("Linear SVM", "linear", dict(C=1.0)),
        ("Polynomial SVM", "poly", dict(degree=3, C=1.0, gamma="scale")),
        ("RBF SVM", "rbf", dict(C=1.0, gamma="scale")),
    ]:
        model = train_kernel(kernel, X_train_s, y_train, **kwargs)
        y_pred = model.predict(X_test_s)
        predictions[name] = y_pred
        results.append(report(name, y_test, y_pred))

    print("\n" + "=" * 60)
    print("BASELINE KERNEL COMPARISON")
    print("=" * 60)
    print(pd.DataFrame([asdict(r) for r in results]).round(4).to_string(index=False))

    grid = tune_rbf(X_train, y_train)  # pipeline scales internally
    y_pred_best = grid.best_estimator_.predict(X_test)
    results.append(report("Optimized RBF", y_test, y_pred_best))

    print("\nFull classification report — Optimized RBF:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    if not args.no_plots:
        plot_class_distribution(df, save_path)
        plot_model_comparison(results, save_path)
        plot_confusion(y_test, y_pred_best, "Optimized RBF SVM - Confusion Matrix", save_path)
        plot_error_comparison(predictions["RBF SVM"], y_pred_best, y_test, save_path)
        plot_precision_recall(results, save_path)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(pd.DataFrame([asdict(r) for r in results]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
