"""Tests for the TF-IDF vectorizer wrapper and training helpers."""

from pathlib import Path
import tempfile

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pytest

from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.utils.tfidf_vectorizer import (
    TfidfVectorizerWrapper,
    combine_title_text,
    split_train_test,
)


@pytest.fixture
def sample_docs():
    return [
        "the president signed a new bill today",
        "breaking news scandal shock outrage",
        "scientists discover new species in amazon",
        "stock market reaches record high amid gains",
        "celebrity couple announces surprise wedding",
    ]


@pytest.fixture
def sample_labels():
    return [1, 0, 1, 1, 0]


def test_wrapper_fit_transform(sample_docs):
    vectorizer = TfidfVectorizerWrapper(max_features=1000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(sample_docs)
    assert vectorizer._fitted
    assert matrix.shape[0] == len(sample_docs)
    assert matrix.shape[1] > 0
    assert vectorizer.n_features == matrix.shape[1]


def test_wrapper_feature_names(sample_docs):
    vectorizer = TfidfVectorizerWrapper(max_features=100)
    vectorizer.fit(sample_docs)
    names = vectorizer.feature_names_
    assert len(names) == vectorizer.n_features
    assert all(isinstance(n, str) for n in names)


def test_wrapper_transform_after_fit(sample_docs):
    vectorizer = TfidfVectorizerWrapper(max_features=1000)
    vectorizer.fit(sample_docs[:3])
    matrix = vectorizer.transform(sample_docs[3:])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == vectorizer.n_features


def test_wrapper_transform_unfitted_raises():
    vectorizer = TfidfVectorizerWrapper()
    with pytest.raises(RuntimeError, match="not been fitted"):
        vectorizer.transform(["some text"])


def test_wrapper_save_and_load(tmp_path, sample_docs):
    vectorizer = TfidfVectorizerWrapper(max_features=1000)
    vectorizer.fit(sample_docs)
    save_path = tmp_path / "tfidf_vectorizer.joblib"
    vectorizer.save(save_path)
    assert save_path.exists()

    loaded = TfidfVectorizerWrapper.load(save_path)
    assert loaded._fitted
    assert loaded.n_features == vectorizer.n_features
    assert loaded.feature_names_ == vectorizer.feature_names_

    # Transform should produce identical results.
    original = vectorizer.transform(sample_docs)
    reloaded = loaded.transform(sample_docs)
    np.testing.assert_array_almost_equal(original.toarray(), reloaded.toarray())


def test_wrapper_save_unfitted_raises():
    vectorizer = TfidfVectorizerWrapper()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
            vectorizer.save(Path(tmp) / "x.joblib")


def test_wrapper_empty_fit_raises():
    vectorizer = TfidfVectorizerWrapper()
    with pytest.raises(ValueError, match="empty"):
        vectorizer.fit([])


def test_combine_title_text():
    titles = ["breaking news", "market update"]
    bodies = ["the president signed a bill", "stocks rose today"]
    combined = combine_title_text(titles, bodies)
    assert combined == ["breaking news the president signed a bill", "market update stocks rose today"]

    with pytest.raises(ValueError, match="same length"):
        combine_title_text(["a"], ["b", "c"])


def test_split_train_test_stratified(sample_docs, sample_labels):
    (
        train_titles,
        test_titles,
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = split_train_test(sample_docs, sample_docs, sample_labels, test_size=0.4, random_state=42)
    assert len(train_titles) + len(test_titles) == len(sample_docs)
    assert len(train_labels) + len(test_labels) == len(sample_labels)
    # Both classes should be represented in each split (stratified).
    assert set(train_labels) == {0, 1}
    assert set(test_labels) == {0, 1}


def test_default_config_matches_plan():
    """Confirm PLAN.md Section 4.B defaults are applied."""
    vectorizer = TfidfVectorizerWrapper()
    assert vectorizer.max_features == 10000
    assert vectorizer.ngram_range == (1, 3)
    assert vectorizer.sublinear_tf is True
    assert isinstance(vectorizer.vectorizer, TfidfVectorizer)
    assert vectorizer.vectorizer.lowercase is False


def test_joblib_roundtrip_preserves_config(tmp_path, sample_docs):
    vectorizer = TfidfVectorizerWrapper(max_features=500, ngram_range=(1, 3), sublinear_tf=True)
    vectorizer.fit(sample_docs)
    path = tmp_path / "vec.joblib"
    vectorizer.save(path)
    loaded = TfidfVectorizerWrapper.load(path)
    assert loaded.max_features == 500
    assert loaded.ngram_range == (1, 3)
    assert loaded.sublinear_tf is True