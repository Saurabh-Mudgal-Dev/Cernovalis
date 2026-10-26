"""Training and tuning of Logistic Regression and Naive Bayes baselines."""

from __future__ import annotations

from typing import Dict, List, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB

DEFAULT_LOGISTIC_C_VALUES = [0.1, 1.0, 10.0]
DEFAULT_LOGISTIC_MAX_ITER = 2000
DEFAULT_NB_ALPHAS = [0.1, 0.5, 1.0]
DEFAULT_RANDOM_STATE = 42

FAKE_LABEL = 0
REAL_LABEL = 1

DEFAULT_LOGISTIC_PATH = "backend/ml/models/saved/logistic_regression.joblib"
DEFAULT_NB_PATH = "backend/ml/models/saved/naive_bayes.joblib"

DEFAULT_RF_N_ESTIMATORS = [100, 300]
DEFAULT_RF_MAX_DEPTH = [None, 20]
DEFAULT_RF_MIN_SAMPLES_SPLIT = [2, 5]
DEFAULT_RF_MAX_FEATURES = "sqrt"
DEFAULT_RF_N_JOBS = -1
DEFAULT_RF_PATH = "backend/ml/models/saved/random_forest.joblib"


def train_logistic_regression(
    X_train,
    y_train: Sequence[int],
    C: float = 1.0,
    max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(
    X_train,
    y_train: Sequence[int],
    alpha: float = 1.0,
) -> MultinomialNB:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y: Sequence[int]) -> Dict[str, object]:
    """Score a fitted model with accuracy, macro F1 and per-class metrics."""
    y = np.asarray(y)
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=[FAKE_LABEL, REAL_LABEL], zero_division=0
    )
    result: Dict[str, object] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            "fake": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
            },
            "real": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
            },
        },
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[FAKE_LABEL, REAL_LABEL]).tolist(),
        "n_samples": int(len(y)),
    }
    return result


def _tune(
    models: Sequence,
    names: List[Dict[str, object]],
    X_val,
    y_val: Sequence[int],
) -> tuple[object, List[Dict[str, object]]]:
    if not models:
        raise ValueError("parameter grid is empty")
    results = [
        {"params": name, "val_metrics": evaluate_model(model, X_val, y_val)}
        for model, name in zip(models, names)
    ]
    best_index = max(range(len(results)), key=lambda i: results[i]["val_metrics"]["macro_f1"])
    return models[best_index], results


def tune_logistic_regression(
    X_train,
    y_train: Sequence[int],
    X_val,
    y_val: Sequence[int],
    C_values: Sequence[float] = DEFAULT_LOGISTIC_C_VALUES,
    max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[LogisticRegression, List[Dict[str, object]]]:
    """Fit every C value on the training split, pick the best by validation macro F1."""
    if not C_values:
        raise ValueError("C_values must not be empty")
    models = [
        train_logistic_regression(X_train, y_train, C=c, max_iter=max_iter, random_state=random_state)
        for c in C_values
    ]
    names = [{"C": c} for c in C_values]
    return _tune(models, names, X_val, y_val)


def tune_naive_bayes(
    X_train,
    y_train: Sequence[int],
    X_val,
    y_val: Sequence[int],
    alphas: Sequence[float] = DEFAULT_NB_ALPHAS,
) -> tuple[MultinomialNB, List[Dict[str, object]]]:
    """Fit every alpha on the training split, pick the best by validation macro F1."""
    if not alphas:
        raise ValueError("alphas must not be empty")
    models = [train_naive_bayes(X_train, y_train, alpha=a) for a in alphas]
    names = [{"alpha": a} for a in alphas]
    return _tune(models, names, X_val, y_val)


def train_random_forest(
    X_train,
    y_train: Sequence[int],
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    max_features: str = DEFAULT_RF_MAX_FEATURES,
    n_jobs: int = DEFAULT_RF_N_JOBS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(
    X_train,
    y_train: Sequence[int],
    X_val,
    y_val: Sequence[int],
    n_estimators: Sequence[int] = DEFAULT_RF_N_ESTIMATORS,
    max_depths: Sequence[int | None] = DEFAULT_RF_MAX_DEPTH,
    min_samples_splits: Sequence[int] = DEFAULT_RF_MIN_SAMPLES_SPLIT,
    n_jobs: int = DEFAULT_RF_N_JOBS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[RandomForestClassifier, List[Dict[str, object]]]:
    """Fit every grid combination on the training split, pick the best by validation macro F1."""
    from itertools import product

    if not n_estimators or not max_depths or not min_samples_splits:
        raise ValueError("random forest grid lists must not be empty")

    models = [
        train_random_forest(
            X_train,
            y_train,
            n_estimators=n_estimators_value,
            max_depth=max_depth_value,
            min_samples_split=min_samples_split_value,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        for n_estimators_value, max_depth_value, min_samples_split_value in product(
            n_estimators, max_depths, min_samples_splits
        )
    ]
    names = [
        {
            "n_estimators": n_estimators_value,
            "max_depth": max_depth_value,
            "min_samples_split": min_samples_split_value,
        }
        for n_estimators_value, max_depth_value, min_samples_split_value in product(
            n_estimators, max_depths, min_samples_splits
        )
    ]
    return _tune(models, names, X_val, y_val)


def save_model(model, path: str) -> str:
    """Persist a fitted model with joblib and return the written path."""
    joblib.dump(model, path)
    return path
