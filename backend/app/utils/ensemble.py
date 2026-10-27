"""Soft-voting ensemble and probability-stacking meta-classifier over the trained models."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

FAKE_LABEL = 0
REAL_LABEL = 1

DEFAULT_META_C_VALUES = [0.1, 1.0, 10.0]
DEFAULT_META_CV_FOLDS = 5
DEFAULT_META_MAX_ITER = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

DEFAULT_META_PATH = "backend/ml/models/saved/meta_classifier.joblib"


def get_probabilities(model, X, verbose: int = 0) -> np.ndarray:
    """Return the positive-class probability for any model (sklearn or Keras)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, -1]
    probs = np.asarray(model.predict(X, verbose=verbose))
    return probs.reshape(len(probs), -1)[:, -1].ravel()


def soft_vote(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Weighted-average per-model probabilities into a single probability vector."""
    if not probabilities:
        raise ValueError("probabilities map must not be empty")
    matrix = np.column_stack([np.asarray(p).ravel() for p in probabilities.values()])
    n = matrix.shape[0]
    if any(np.asarray(p).ravel().shape[0] != n for p in probabilities.values()):
        raise ValueError("all probability arrays must have the same length")
    if weights is None:
        weights = {name: 1.0 for name in probabilities}
    total = sum(float(weights.get(name, 0.0)) for name in probabilities)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w = np.asarray(
        [float(weights.get(name, 0.0)) for name in probabilities], dtype=np.float64
    ) / total
    return (matrix * w).sum(axis=1)


def weights_from_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    """Turn validation scores (e.g. macro F1) into normalized voting weights."""
    total = sum(max(float(score), 0.0) for score in scores.values())
    if total <= 0:
        raise ValueError("scores must sum to a positive value")
    return {name: max(float(score), 0.0) / total for name, score in scores.items()}


def evaluate_probabilities(
    y_true: Sequence[int],
    probs: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, object]:
    """Accuracy, macro F1, per-class metrics and confusion matrix from a probability vector."""
    y = np.asarray(y_true)
    y_pred = (np.asarray(probs).ravel() >= threshold).astype(int)
    if y.shape != y_pred.shape:
        raise ValueError("y_true and probs must align")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=[FAKE_LABEL, REAL_LABEL], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            "fake": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])},
            "real": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])},
        },
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[FAKE_LABEL, REAL_LABEL]).tolist(),
        "n_samples": int(len(y)),
    }


def tune_meta_classifier(
    val_probs: np.ndarray,
    y_val: Sequence[int],
    C_values: Sequence[float] = DEFAULT_META_C_VALUES,
    cv_folds: int = DEFAULT_META_CV_FOLDS,
    max_iter: int = DEFAULT_META_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[LogisticRegression, List[Dict[str, object]]]:
    """Stack val probabilities and pick the logistic meta-learner's C by cross-validated macro F1."""
    from itertools import product

    del product

    X = np.asarray(val_probs, dtype=np.float64)
    y = np.asarray(y_val)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("val_probs must be a 2-D matrix of per-model probabilities")
    if len(X) != len(y):
        raise ValueError("val_probs and y_val must align")
    if len(set(y.tolist())) < 2:
        raise ValueError("meta-classifier needs both classes in y_val")
    if not C_values:
        raise ValueError("C_values must not be empty")

    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    models, results = [], []
    for c in C_values:
        candidate = LogisticRegression(
            C=c,
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        )
        cv_scores = cross_val_score(
            candidate, X, y, cv=splitter, scoring="f1_macro"
        )
        models.append(candidate)
        results.append({
            "params": {"C": c},
            "cv_macro_f1": float(np.mean(cv_scores)),
            "cv_fold_macro_f1": [float(s) for s in cv_scores],
        })
    best_index = max(range(len(results)), key=lambda i: results[i]["cv_macro_f1"])
    best = models[best_index]
    best.fit(X, y)
    return best, results


def stacked_matrix(probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
    """Column-stack named probability arrays into the meta-classifier's feature matrix."""
    if not probabilities:
        raise ValueError("probabilities map must not be empty")
    matrix = np.column_stack([np.asarray(p).ravel() for p in probabilities.values()])
    lengths = {np.asarray(p).ravel().shape[0] for p in probabilities.values()}
    if len(lengths) != 1:
        raise ValueError("all probability arrays must have the same length")
    return matrix


def save_meta(model, path: str = DEFAULT_META_PATH) -> str:
    """Persist the fitted meta-classifier with joblib."""
    joblib.dump(model, path)
    return path


def load_meta(path: str = DEFAULT_META_PATH) -> LogisticRegression:
    """Load a model previously written by save_meta."""
    if path.endswith(".keras"):
        raise ValueError("meta-classifier path must be a joblib file, not .keras")
    return joblib.load(path)
