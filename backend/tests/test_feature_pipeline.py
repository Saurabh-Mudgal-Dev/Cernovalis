"""Tests for the unified feature pipeline."""

import numpy as np
import pytest
from scipy.sparse import issparse

from backend.app.utils.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_titles():
    return [
        "breaking news scandal",
        "market update today",
        "president signs new bill",
        "scientists discover species",
    ]


@pytest.fixture
def sample_texts():
    return [
        "the president signed a shocking new bill today",
        "stocks rose on wall street amid gains",
        "legislation passes congress with bipartisan support",
        "new species found in amazon rainforest by researchers",
    ]


def test_pipeline_fit_transform(sample_titles, sample_texts):
    pipeline = FeaturePipeline(max_features=1000)
    matrix = pipeline.fit_transform(sample_titles, sample_texts)
    assert pipeline._fitted
    assert issparse(matrix)
    assert matrix.shape[0] == len(sample_titles)
    assert matrix.shape[1] > 0
    assert pipeline.n_features == matrix.shape[1]


def test_pipeline_transform_after_fit(sample_titles, sample_texts):
    pipeline = FeaturePipeline(max_features=1000)
    pipeline.fit(sample_titles[:2], sample_texts[:2])
    matrix = pipeline.transform(sample_titles[2:], sample_texts[2:])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == pipeline.n_features


def test_pipeline_transform_unfitted_raises(sample_titles, sample_texts):
    pipeline = FeaturePipeline()
    with pytest.raises(RuntimeError, match="not been fitted"):
        pipeline.transform(sample_titles, sample_texts)


def test_pipeline_save_and_load(tmp_path, sample_titles, sample_texts):
    pipeline = FeaturePipeline(max_features=1000)
    pipeline.fit(sample_titles, sample_texts)
    save_path = tmp_path / "pipeline.joblib"
    pipeline.save(
        save_path,
        tmp_path / "tfidf_vectorizer.joblib",
        tmp_path / "feature_scaler.joblib",
    )
    assert save_path.exists()

    loaded = FeaturePipeline.load(save_path)
    assert loaded._fitted
    assert loaded.n_features == pipeline.n_features
    assert loaded.feature_names == pipeline.feature_names

    original = pipeline.transform(sample_titles, sample_texts)
    reloaded = loaded.transform(sample_titles, sample_texts)
    np.testing.assert_array_almost_equal(
        original.toarray(), reloaded.toarray()
    )


def test_pipeline_save_unfitted_raises(tmp_path):
    pipeline = FeaturePipeline()
    with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
        pipeline.save(tmp_path / "x.joblib")


def test_pipeline_empty_fit_raises():
    pipeline = FeaturePipeline()
    with pytest.raises(ValueError, match="empty"):
        pipeline.fit([], [])


def test_pipeline_mismatched_lengths_raises():
    pipeline = FeaturePipeline()
    with pytest.raises(ValueError, match="same length"):
        pipeline.fit(["a"], ["b", "c"])


def test_pipeline_default_config():
    pipeline = FeaturePipeline()
    assert pipeline.tfidf.max_features == 10000
    assert pipeline.tfidf.ngram_range == (1, 3)
    assert pipeline.tfidf.sublinear_tf is True


def test_pipeline_feature_names_count(sample_titles, sample_texts):
    pipeline = FeaturePipeline(max_features=1000)
    pipeline.fit(sample_titles, sample_texts)
    names = pipeline.feature_names
    assert len(names) == pipeline.n_features
    assert all(isinstance(n, str) for n in names)
    assert len(set(names)) == len(names)


def test_pipeline_joblib_roundtrip_preserves_config(tmp_path, sample_titles, sample_texts):
    pipeline = FeaturePipeline(max_features=500, ngram_range=(1, 2), sublinear_tf=False)
    pipeline.fit(sample_titles, sample_texts)
    path = tmp_path / "pipeline.joblib"
    pipeline.save(
        path,
        tmp_path / "tfidf_vectorizer.joblib",
        tmp_path / "feature_scaler.joblib",
    )
    loaded = FeaturePipeline.load(path)
    assert loaded.tfidf.max_features == 500
    assert loaded.tfidf.ngram_range == (1, 2)
    assert loaded.tfidf.sublinear_tf is False


def test_pipeline_save_persists_vectorizer_and_scaler(
    tmp_path, sample_titles, sample_texts
):
    pipeline = FeaturePipeline(max_features=1000)
    pipeline.fit(sample_titles, sample_texts)

    pipeline_path = tmp_path / "pipeline.joblib"
    vectorizer_path = tmp_path / "tfidf_vectorizer.joblib"
    scaler_path = tmp_path / "feature_scaler.joblib"

    pipeline.save(pipeline_path, vectorizer_path, scaler_path)

    assert pipeline_path.exists()
    assert vectorizer_path.exists()
    assert scaler_path.exists()

    from backend.app.utils.feature_scaler import FeatureScaler

    loaded_scaler = FeatureScaler.load(scaler_path)
    assert loaded_scaler._fitted
    assert loaded_scaler.n_features == len(pipeline.feature_names) - pipeline.tfidf.n_features