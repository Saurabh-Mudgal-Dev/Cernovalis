"""Tests for the Logistic Regression, Naive Bayes, Random Forest and XGBoost models."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from backend.app.utils.baselines import (
    DEFAULT_LOGISTIC_C_VALUES,
    DEFAULT_NB_ALPHAS,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MIN_SAMPLES_SPLIT,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_N_ESTIMATORS,
    evaluate_model,
    save_model,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
    tune_logistic_regression,
    tune_naive_bayes,
    tune_random_forest,
    tune_xgboost,
)
from backend.app.utils.dataset_split import stratified_split

FAKE_LABEL = 0
REAL_LABEL = 1


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


def test_train_random_forest_default_params(separable_features):
    X, y = separable_features
    model = train_random_forest(X, y, n_estimators=10, n_jobs=1)
    assert model.n_estimators == 10
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_train_random_forest_reproducible(separable_features):
    X, y = separable_features
    model = train_random_forest(X, y, n_estimators=10, random_state=42, n_jobs=1)
    clone = train_random_forest(X, y, n_estimators=10, random_state=42, n_jobs=1)
    np.testing.assert_array_equal(model.predict(X), clone.predict(X))


def test_tune_random_forest_picks_grid_member(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    model, results = tune_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=[10],
        max_depths=[None, 3],
        min_samples_splits=[2],
        n_jobs=1,
    )
    assert len(results) == 2
    assert set(r["params"]["max_depth"] for r in results) == {None, 3}
    params = model.get_params()
    assert params["max_depth"] in (None, 3)
    assert params["n_estimators"] == 10
    assert params["min_samples_split"] == 2
    for r in results:
        assert r["val_metrics"]["n_samples"] == len(y_val)


def test_tune_random_forest_grid_size(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    _, results = tune_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=[10],
        max_depths=[None, 3],
        min_samples_splits=[2, 5],
        n_jobs=1,
    )
    assert len(results) == 4


def test_tune_random_forest_is_deterministic(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    m1, r1 = tune_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=[10], max_depths=[3], min_samples_splits=[2], n_jobs=1,
    )
    m2, r2 = tune_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=[10], max_depths=[3], min_samples_splits=[2], n_jobs=1,
    )
    assert m1.get_params()["max_depth"] == m2.get_params()["max_depth"]
    assert r1[0]["val_metrics"]["accuracy"] == r2[0]["val_metrics"]["accuracy"]


def test_tune_random_forest_empty_grid_raises(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    with pytest.raises(ValueError):
        tune_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=[], max_depths=[3], min_samples_splits=[2],
        )


def test_default_rf_grids_exist():
    assert DEFAULT_RF_N_ESTIMATORS
    assert DEFAULT_RF_MAX_DEPTH
    assert DEFAULT_RF_MIN_SAMPLES_SPLIT


def test_save_random_forest_roundtrip(separable_features, tmp_path):
    X, y = separable_features
    model = train_random_forest(X, y, n_estimators=10, n_jobs=1)
    path = save_model(model, tmp_path / "rf.joblib")
    import joblib

    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_train_xgboost_default_params(separable_features):
    X, y = separable_features
    model = train_xgboost(X, y, n_estimators=10, n_jobs=1)
    assert model.n_estimators == 10
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(np.unique(predictions)) <= {0, 1}


def test_train_xgboost_reproducible(separable_features):
    X, y = separable_features
    model = train_xgboost(X, y, n_estimators=10, random_state=42, n_jobs=1)
    clone = train_xgboost(X, y, n_estimators=10, random_state=42, n_jobs=1)
    np.testing.assert_array_equal(model.predict(X), clone.predict(X))


def test_tune_xgboost_picks_grid_member(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    model, results = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=[10],
        max_depths=[2, 3],
        learning_rates=[0.1],
        n_jobs=1,
    )
    assert len(results) == 2
    assert set(r["params"]["max_depth"] for r in results) == {2, 3}
    assert model.get_params()["max_depth"] in {2, 3}
    assert model.get_params()["n_estimators"] == 10
    for r in results:
        assert r["val_metrics"]["n_samples"] == len(y_val)


def test_tune_xgboost_grid_size(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    _, results = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=[10],
        max_depths=[2, 3],
        learning_rates=[0.1, 0.5],
        n_jobs=1,
    )
    assert len(results) == 4


def test_tune_xgboost_is_deterministic(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    m1, r1 = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=[10], max_depths=[2], learning_rates=[0.1], n_jobs=1,
    )
    m2, r2 = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=[10], max_depths=[2], learning_rates=[0.1], n_jobs=1,
    )
    assert m1.get_params()["max_depth"] == m2.get_params()["max_depth"]
    assert r1[0]["val_metrics"]["accuracy"] == r2[0]["val_metrics"]["accuracy"]


def test_tune_xgboost_empty_grid_raises(separable_features):
    X, y = separable_features
    X_train, X_val = X[:50], X[50:]
    y_train, y_val = y[:50], y[50:]
    with pytest.raises(ValueError):
        tune_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=[], max_depths=[3], learning_rates=[0.1],
        )


def test_default_xgb_grids_exist():
    assert DEFAULT_XGB_N_ESTIMATORS
    assert DEFAULT_XGB_MAX_DEPTH


def test_save_xgboost_roundtrip(separable_features, tmp_path):
    X, y = separable_features
    model = train_xgboost(X, y, n_estimators=10, n_jobs=1)
    path = save_model(model, tmp_path / "xgb.joblib")
    import joblib

    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
