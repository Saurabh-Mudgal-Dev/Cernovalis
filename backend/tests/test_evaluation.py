"""Tests for the evaluation utilities: curves, confusion matrices, feature importance, plots."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from backend.app.utils.evaluation import (
    DEFAULT_TOP_K,
    confusion,
    feature_importance,
    plot_confusion,
    plot_feature_importance,
    plot_pr,
    plot_roc,
    pr_points,
    resample_curve,
    roc_points,
    row_normalized_confusion,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def separable():
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    y_pred = [0, 0, 0, 1, 1, 1]
    return y_true, scores, y_pred


def test_roc_auc_perfect_separation(separable):
    y_true, scores, _ = separable
    result = roc_points(y_true, scores)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert len(result["fpr"]) == len(result["tpr"])


def test_roc_auc_inverted_scores(separable):
    y_true, scores, _ = separable
    inverted = [1 - s for s in scores]
    assert roc_points(y_true, inverted)["roc_auc"] == pytest.approx(0.0)


def test_curve_length_mismatch_raises(separable):
    y_true, _, _ = separable
    with pytest.raises(ValueError):
        roc_points(y_true, [0.5])
    with pytest.raises(ValueError):
        pr_points(y_true, [0.5])


def test_average_precision_perfect(separable):
    y_true, scores, _ = separable
    result = pr_points(y_true, scores)
    assert result["average_precision"] == pytest.approx(1.0)
    assert len(result["precision"]) == len(result["recall"])


def test_confusion_label_order(separable):
    y_true, _, y_pred = separable
    assert confusion(y_true, y_pred) == [[3, 0], [0, 3]]
    # one real article misread as fake
    assert confusion(y_true, [0, 0, 0, 0, 1, 1]) == [[3, 0], [1, 2]]


def test_row_normalized_confusion(separable):
    y_true, _, _ = separable
    matrix = row_normalized_confusion(y_true, [0, 0, 0, 0, 1, 1])
    assert matrix[0] == pytest.approx([1.0, 0.0])
    assert matrix[1] == pytest.approx([1 / 3, 2 / 3])


def test_resample_curve_interpolates_onto_grid():
    x = [0.0, 0.5, 1.0]
    y = [0.0, 0.25, 1.0]
    grid, sampled = resample_curve(x, y)
    assert len(grid) == len(sampled)
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert sampled[0] == pytest.approx(0.0)
    assert sampled[-1] == pytest.approx(1.0)
    mid = sampled[len(sampled) // 2]
    assert 0.0 < mid < 1.0


def test_resample_curve_deduplicates_flat_x():
    grid, sampled = resample_curve([0.5, 0.5, 0.5], [1.0, 0.5, 0.0])
    assert len(grid) == len(sampled)
    # a constant source maps every grid point to the single y value
    assert all(value == pytest.approx(1.0) for value in sampled)


def test_feature_importance_logistic_top_k():
    rng = np.random.RandomState(0)
    X = rng.rand(80, 5)
    y = (X[:, 2] + X[:, 4] > 1.0).astype(int)
    model = LogisticRegression().fit(X, y)
    ranked = feature_importance(model, top_k=3)
    assert len(ranked) == 3
    assert ranked[0]["feature"] in {"f2", "f4"}
    values = [entry["importance"] for entry in ranked]
    assert values == sorted(values, reverse=True)


def test_feature_importance_uses_model_feature_names():
    rng = np.random.RandomState(1)
    X = rng.rand(60, 3)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)
    ranked = feature_importance(model, ["alpha", "beta", "gamma"], top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["feature"] == "alpha"


def test_feature_importance_random_forest():
    rng = np.random.RandomState(2)
    X = rng.rand(100, 4)
    y = (X[:, 1] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    ranked = feature_importance(model, top_k=None)
    assert len(ranked) == 4
    assert ranked[0]["feature"] == "f1"


def test_feature_importance_naive_bayes():
    counts = np.array([[5, 0, 1], [4, 0, 2], [0, 5, 1], [1, 4, 2]])
    labels = [0, 0, 1, 1]
    model = MultinomialNB().fit(counts, labels)
    ranked = feature_importance(model, ["news", "sport", "noise"])
    assert len(ranked) == 3
    # "sport" never appears in the fake class, so it separates the classes hardest
    assert ranked[0]["feature"] == "sport"
    assert ranked[-1]["feature"] == "noise"


def test_feature_importance_unknown_model_returns_empty():
    assert feature_importance(object()) == []


def test_feature_importance_name_mismatch_raises():
    model = LogisticRegression().fit(np.random.rand(20, 3), [0, 1] * 10)
    with pytest.raises(ValueError):
        feature_importance(model, ["only", "two"])


def test_plot_confusion_writes_png(tmp_path, separable):
    y_true, _, y_pred = separable
    path = tmp_path / "cm.png"
    result = plot_confusion(y_true, y_pred, path)
    assert result == path
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_plot_confusion_normalized_writes_png(tmp_path, separable):
    y_true, _, y_pred = separable
    path = plot_confusion(y_true, y_pred, tmp_path / "cm_norm.png", normalize=True)
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_plot_roc_and_pr_write_pngs(tmp_path, separable):
    y_true, scores, _ = separable
    roc_path = plot_roc({"model": roc_points(y_true, scores)}, tmp_path / "roc.png")
    pr_path = plot_pr(
        {"model": pr_points(y_true, scores)}, tmp_path / "pr.png"
    )
    assert roc_path.read_bytes()[:8] == PNG_MAGIC
    assert pr_path.read_bytes()[:8] == PNG_MAGIC


def test_plot_roc_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_roc({}, tmp_path / "roc.png")


def test_plot_feature_importance_writes_png(tmp_path):
    entries = [{"feature": f"tok{i}", "importance": 1.0 / (i + 1)} for i in range(5)]
    path = plot_feature_importance(entries, tmp_path / "fi.png")
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_plot_feature_importance_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_feature_importance([], tmp_path / "fi.png")


def test_default_top_k_is_exported():
    assert DEFAULT_TOP_K > 0
