"""Tests for the Logistic Regression and Naive Bayes baselines."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from backend.app.utils.baselines import (
    DEFAULT_LOGISTIC_C_VALUES,
    DEFAULT_NB_ALPHAS,
    FAKE_LABEL,
    REAL_LABEL,
    evaluate_model,
    save_model,
    train_logistic_regression,
    train_naive_bayes,
    tune_logistic_regression,
    tune_naive_bayes,
)
from backend.app.utils.dataset_split import stratified_split


@pytest.fixture
def separable_features():
    """Sparse matrix with an informative 'fake' feature and an informative 'real' feature."""
    rows = []
    labels = []
    rng = np.random.default_rng(42)
    for _ in range(40):
        rows.append([1.0, 0.0])
        labels.append(FAKE_LABEL)
    for _ in range(40):
        rows.append([0.0, 1.0])
        labels.append(REAL_LABEL)
    X = csr_matrix(np.array(rows))
    # Shuffle the labels against the feature rows using a FIXED order so the
    # split remains deterministic for the tests below.
    order = rng.permutation(len(labels))
    X = X[order]
    labels = [labels[i] for i in order]
    return X, labels


def test_train_logistic_regression_default_params(separable_features):
    X, y = separable_features
    model = train_logistic_regression(X, y)
    assert model.C == 1.0
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_train_naive_bayes_default_params(separable_features):
    X, y = separable_features
    model = train_naive_bayes(X, y)
    assert model.alpha == 1.0
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_evaluate_model_perfect_predictions(separable_features):
    X, y = separable_features
    model = train_logistic_regression(X, y)
    metrics = evaluate_model(model, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["fake"]["f1"] == 1.0
    assert metrics["per_class"]["real"]["f1"] == 1.0
    assert metrics["n_samples"] == len(y)
    assert metrics["per_class"]["fake"]["precision"] == 1.0
    assert metrics["per_class"]["fake"]["recall"] == 1.0
    assert metrics["per_class"]["real"]["precision"] == 1.0
    assert metrics["per_class"]["real"]["recall"] == 1.0


def test_tune_logistic_regression_picks_grid_member(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    model, results = tune_logistic_regression(X_train, y_train, X_val, y_val)
    assert {r["params"]["C"] for r in results} == set(DEFAULT_LOGISTIC_C_VALUES)
    assert model.C in DEFAULT_LOGISTIC_C_VALUES
    for r in results:
        assert r["val_metrics"]["n_samples"] == len(y_val)


def test_tune_naive_bayes_picks_grid_member(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    model, results = tune_naive_bayes(X_train, y_train, X_val, y_val)
    assert {r["params"]["alpha"] for r in results} == set(DEFAULT_NB_ALPHAS)
    assert model.alpha in DEFAULT_NB_ALPHAS
    for r in results:
        assert r["val_metrics"]["n_samples"] == len(y_val)


def test_tune_is_deterministic(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    m1, r1 = tune_logistic_regression(X_train, y_train, X_val, y_val)
    m2, r2 = tune_logistic_regression(X_train, y_train, X_val, y_val)
    assert m1.C == m2.C
    assert r1[0]["val_metrics"]["accuracy"] == r2[0]["val_metrics"]["accuracy"]


def test_tune_empty_grid_raises(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    with pytest.raises(ValueError):
        tune_logistic_regression(X_train, y_train, X_val, y_val, C_values=[])
    with pytest.raises(ValueError):
        tune_naive_bayes(X_train, y_train, X_val, y_val, alphas=[])


def test_save_model_roundtrip(separable_features, tmp_path):
    X, y = separable_features
    model = train_naive_bayes(X, y)
    path = save_model(model, tmp_path / "nb.joblib")
    import joblib

    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_stratified_split_produces_nonempty_val_and_test():
    labels = [FAKE_LABEL] * 100 + [REAL_LABEL] * 100
    split = stratified_split(labels)
    train, val, test = split.subset(labels)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
