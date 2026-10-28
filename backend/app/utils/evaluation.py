"""Confusion matrices, ROC-AUC, precision-recall curves and feature-importance artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FAKE_LABEL = 0
REAL_LABEL = 1
CLASS_NAMES = ["fake", "real"]
DEFAULT_TOP_K = 25

FULL_GRID = np.linspace(0.0, 1.0, 201)


def roc_points(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, object]:
    """ROC curve points plus the scalar ROC-AUC for one set of positive-class scores."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(roc_auc_score(y, s)),
    }


def pr_points(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, object]:
    """Precision-recall curve points plus average precision for one score vector."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    precision, recall, _ = precision_recall_curve(y, s)
    return {
        "precision": precision,
        "recall": recall,
        "average_precision": float(average_precision_score(y, s)),
    }


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> List[List[int]]:
    """Raw 2x2 confusion matrix over [FAKE, REAL] labels."""
    return confusion_matrix(
        np.asarray(y_true), np.asarray(y_pred), labels=[FAKE_LABEL, REAL_LABEL]
    ).tolist()


def row_normalized_confusion(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> List[List[float]]:
    """Confusion matrix with each true-class row scaled to sum to 1."""
    counts = np.asarray(
        confusion_matrix(
            np.asarray(y_true), np.asarray(y_pred), labels=[FAKE_LABEL, REAL_LABEL]
        ),
        dtype=float,
    )
    row_sums = counts.sum(axis=1, keepdims=True)
    denominator = np.where(row_sums == 0, 1.0, row_sums)
    return (counts / denominator).tolist()


def feature_importance(
    model, feature_names: Sequence[str] | None = None, top_k: int | None = DEFAULT_TOP_K
) -> List[Dict[str, object]]:
    """Rank a fitted model's per-feature importances, handling coef_, feature_importances_ and NB log-probs."""
    importances = None
    if hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).ravel()
    elif hasattr(model, "feature_log_prob_"):
        log_probs = np.asarray(model.feature_log_prob_)
        if log_probs.shape[0] == 2:
            importances = np.abs(log_probs[1] - log_probs[0])
    if importances is None:
        return []

    n_features = int(importances.shape[0])
    if feature_names is None:
        names = [f"f{index}" for index in range(n_features)]
    else:
        names = [str(name) for name in feature_names]
        if len(names) != n_features:
            raise ValueError(
                f"{len(names)} feature names for {n_features} importances"
            )

    order = np.argsort(importances)[::-1]
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    return [
        {"feature": names[index], "importance": float(importances[index])}
        for index in order
    ]


def resample_curve(x: Sequence[float], y: Sequence[float], grid: np.ndarray = FULL_GRID):
    """Interpolate a monotone-x curve onto a fixed grid so JSON stays compact."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0:
        return grid.tolist(), np.zeros(grid.shape).tolist()
    keep = np.concatenate(([True], np.diff(x_arr) > 0))
    x_arr, y_arr = x_arr[keep], y_arr[keep]
    if x_arr.size == 1:
        return grid.tolist(), np.full(grid.shape, y_arr[0]).tolist()
    return grid.tolist(), np.interp(grid, x_arr, y_arr).tolist()


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    path: Path | str,
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> Path:
    """Render a 2x2 fake/real confusion matrix heatmap and save it as a PNG."""
    plt = _pyplot()
    if normalize:
        matrix = np.asarray(row_normalized_confusion(y_true, y_pred), dtype=float)
    else:
        matrix = np.asarray(confusion(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    threshold = matrix.max() / 2 if matrix.size else 0.0
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            label = f"{value:.2f}" if normalize else str(int(value))
            ax.text(
                col, row, label, ha="center", va="center",
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _save_figure(fig, path)


def plot_roc(curves: Dict[str, Dict[str, object]], path: Path | str) -> Path:
    """Overlay ROC curves for several named models on one figure."""
    if not curves:
        raise ValueError("no curves to plot")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, curve in sorted(curves.items()):
        ax.plot(
            curve["fpr"], curve["tpr"],
            label=f"{name} (AUC = {curve['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set(
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    ax.legend(loc="lower right", fontsize=8)
    return _save_figure(fig, path)


def plot_pr(curves: Dict[str, Dict[str, object]], path: Path | str) -> Path:
    """Overlay precision-recall curves for several named models on one figure."""
    if not curves:
        raise ValueError("no curves to plot")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, curve in sorted(curves.items()):
        ax.plot(
            curve["recall"], curve["precision"],
            label=f"{name} (AP = {curve['average_precision']:.4f})",
        )
    ax.set(
        title="Precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    ax.legend(loc="lower left", fontsize=8)
    return _save_figure(fig, path)


def plot_feature_importance(
    importances: Sequence[Dict[str, object]], path: Path | str, title: str = "Feature importance"
) -> Path:
    """Render a horizontal bar chart of ranked feature importances."""
    entries = list(importances)
    if not entries:
        raise ValueError("no importances to plot")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(entries) + 1)))
    labels = [str(entry["feature"]) for entry in entries][::-1]
    values = [float(entry["importance"]) for entry in entries][::-1]
    ax.barh(labels, values)
    ax.set_xlabel("importance")
    ax.set_title(title)
    return _save_figure(fig, path)
