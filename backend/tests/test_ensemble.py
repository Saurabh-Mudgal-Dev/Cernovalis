"""Tests for the soft-voting ensemble and meta-classifier stacking helpers."""

from pathlib import Path

import numpy as np
import pytest

from backend.app.utils.baselines import evaluate_model
from backend.app.utils.ensemble import (
    DEFAULT_META_C_VALUES,
    DEFAULT_META_PATH,
    FAKE_LABEL,
    REAL_LABEL,
    evaluate_probabilities,
    get_probabilities,
    load_meta,
    save_meta,
    soft_vote,
    stacked_matrix,
    tune_meta_classifier,
    weights_from_scores,
)


class FixedProba:
    """Minimal sklearn-like model returning a canned positive-class probability."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64).ravel()

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


class FixedKeras:
    """Minimal Keras-like model exposing only predict()."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64).ravel()

    def predict(self, X, verbose=0):
        return self.p.reshape(-1, 1)


def test_soft_vote_uniform_mean():
    probs = {"a": np.array([0.8, 0.2]), "b": np.array([0.2, 0.6])}
    np.testing.assert_array_almost_equal(soft_vote(probs), np.array([0.5, 0.4]))


def test_soft_vote_weighted_normalizes_weights():
    probs = {"a": np.array([0.8]), "b": np.array([0.2])}
    voted = soft_vote(probs, weights={"a": 3.0, "b": 1.0})
    np.testing.assert_array_almost_equal(voted, np.array([0.65]))


def test_soft_vote_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        soft_vote({})
    with pytest.raises(ValueError):
        soft_vote({"a": np.array([0.5])}, weights={"a": 0.0})
    with pytest.raises(ValueError):
        soft_vote({"a": np.array([0.5]), "b": np.array([0.3, 0.3])})


def test_weights_from_scores_normalizes_proportionally():
    weights = weights_from_scores({"a": 0.9, "b": 0.1})
    np.testing.assert_array_almost_equal(
        np.array([weights["a"], weights["b"]]), np.array([0.9, 0.1])
    )
    assert weights_from_scores({"a": 4.0, "b": 0.0})["b"] == 0.0
    with pytest.raises(ValueError):
        weights_from_scores({"a": 0.0})


def test_evaluate_probabilities_structure_and_perfect_split():
    y = [0, 1, 0, 1]
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = evaluate_probabilities(y, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["fake"]["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["n_samples"] == 4
    with pytest.raises(ValueError):
        evaluate_probabilities(y, probs[:3])


def test_get_probabilities_sklike_and_keras_paths():
    X = np.zeros((3, 1))
    np.testing.assert_array_almost_equal(
        get_probabilities(FixedProba([0.1, 0.5, 0.9]), X), np.array([0.1, 0.5, 0.9])
    )
    np.testing.assert_array_almost_equal(
        get_probabilities(FixedKeras([0.1, 0.5, 0.9]), X), np.array([0.1, 0.5, 0.9])
    )


def test_stacked_matrix_shapes_and_validation():
    matrix = stacked_matrix({
        "a": np.array([0.1, 0.4]),
        "b": np.array([0.9, 0.6]),
    })
    assert matrix.shape == (2, 2)
    np.testing.assert_array_equal(matrix[:, 0], np.array([0.1, 0.4]))
    with pytest.raises(ValueError):
        stacked_matrix({})
    with pytest.raises(ValueError):
        stacked_matrix({"a": np.array([0.1]), "b": np.array([0.2, 0.3])})


def test_tune_meta_classifier_selects_from_grid():
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([FAKE_LABEL, REAL_LABEL] * (n // 2))
    clean = np.where(y == REAL_LABEL, 0.85, 0.15)
    noisy = np.clip(clean + rng.normal(0, 0.15, size=n), 0.0, 1.0)
    val_probs = np.column_stack([clean + rng.normal(0, 0.03, size=n), noisy])

    best, results = tune_meta_classifier(val_probs, y)
    assert len(results) == len(DEFAULT_META_C_VALUES)
    assert [r["params"]["C"] for r in results] == DEFAULT_META_C_VALUES
    assert all(0.0 <= r["cv_macro_f1"] <= 1.0 for r in results)
    best_entry = max(results, key=lambda r: r["cv_macro_f1"])
    assert np.isclose(
        best_entry["cv_macro_f1"],
        max(r["cv_macro_f1"] for r in results),
    )
    assert getattr(best, "C") == DEFAULT_META_C_VALUES[
        int(np.argmax([r["cv_macro_f1"] for r in results]))
    ]
    assert bool(np.isfinite(best.predict_proba(val_probs)).all())


def test_tune_meta_classifier_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        tune_meta_classifier(np.zeros((4, 1)), [0, 0, 0, 0])
    with pytest.raises(ValueError):
        tune_meta_classifier(np.zeros((4, 1)), [0, 1, 0, 1][:3])
    with pytest.raises(ValueError):
        tune_meta_classifier(np.zeros((4, 1)), [0, 1, 0, 1], C_values=[])


def test_meta_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    y = [0, 1] * 10
    val_probs = np.column_stack([
        np.where(np.array(y) == 1, 0.8, 0.2),
        np.where(np.array(y) == 1, 0.75, 0.25),
    ])
    best, _ = tune_meta_classifier(val_probs, y)
    path = tmp_path / "meta.joblib"
    written = save_meta(best, str(path))
    assert Path(written).exists()
    loaded = load_meta(written)
    np.testing.assert_array_almost_equal(
        loaded.predict_proba(val_probs), best.predict_proba(val_probs)
    )
    with pytest.raises(ValueError):
        load_meta(str(tmp_path / "meta.keras"))


def test_metrics_format_matches_baselines():
    y = [0, 1, 0, 1]
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    probs_metrics = evaluate_probabilities(y, probs)
    class Dummy:
        @staticmethod
        def predict(X):
            return (np.asarray(X).ravel() >= 0.5).astype(int)
    baseline_metrics = evaluate_model(
        Dummy, probs.reshape(-1, 1), y
    )
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "n_samples"):
        assert probs_metrics[key] == baseline_metrics[key]


def test_defaults_are_sane():
    assert DEFAULT_META_C_VALUES == [0.1, 1.0, 10.0]
    assert DEFAULT_META_PATH == "backend/ml/models/saved/meta_classifier.joblib"
