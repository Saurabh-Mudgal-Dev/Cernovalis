"""Tests for the feature scaler."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from backend.app.utils.feature_scaler import FeatureScaler


@pytest.fixture
def sample_matrix():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )


def test_scaler_fit_transform(sample_matrix):
    scaler = FeatureScaler()
    result = scaler.fit_transform(sample_matrix, ["a", "b", "c"])
    assert scaler._fitted
    assert scaler.n_features == 3
    # StandardScaler: mean 0, std 1 per feature
    assert np.allclose(result.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(result.std(axis=0), 1.0, atol=1e-10)


def test_scaler_transform_after_fit(sample_matrix):
    scaler = FeatureScaler()
    scaler.fit(sample_matrix[:2], ["a", "b", "c"])
    result = scaler.transform(sample_matrix[2:])
    assert result.shape == (2, 3)


def test_scaler_transform_unfitted_raises(sample_matrix):
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError, match="not been fitted"):
        scaler.transform(sample_matrix)


def test_scaler_save_and_load(tmp_path, sample_matrix):
    scaler = FeatureScaler()
    scaler.fit(sample_matrix, ["a", "b", "c"])
    save_path = tmp_path / "feature_scaler.joblib"
    scaler.save(save_path)
    assert save_path.exists()

    loaded = FeatureScaler.load(save_path)
    assert loaded._fitted
    assert loaded.n_features == scaler.n_features

    original = scaler.transform(sample_matrix)
    reloaded = loaded.transform(sample_matrix)
    np.testing.assert_array_almost_equal(original, reloaded)


def test_scaler_save_unfitted_raises(tmp_path):
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
        scaler.save(tmp_path / "x.joblib")


def test_scaler_empty_fit_raises():
    scaler = FeatureScaler()
    with pytest.raises(ValueError, match="empty"):
        scaler.fit(np.array([]).reshape(0, 3))


def test_scaler_means_and_scales(sample_matrix):
    scaler = FeatureScaler()
    scaler.fit(sample_matrix, ["a", "b", "c"])
    expected = StandardScaler().fit(sample_matrix)
    np.testing.assert_array_almost_equal(scaler.means, expected.mean_)
    np.testing.assert_array_almost_equal(scaler.scales, expected.scale_)


def test_scaler_joblib_roundtrip_preserves_config(tmp_path, sample_matrix):
    scaler = FeatureScaler()
    scaler.fit(sample_matrix, ["a", "b", "c"])
    path = tmp_path / "scaler.joblib"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    assert loaded.n_features == 3
    np.testing.assert_array_almost_equal(loaded.means, scaler.means)
    np.testing.assert_array_almost_equal(loaded.scales, scaler.scales)