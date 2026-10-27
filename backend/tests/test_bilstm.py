"""Tests for the BiLSTM tokenization, GloVe pipeline and model helpers."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from backend.app.utils.baselines import evaluate_model
from backend.app.utils.bilstm import (
    DEFAULT_BILSTM_PATH,
    DEFAULT_DROPOUTS,
    DEFAULT_LSTM_UNITS,
    DEFAULT_MAX_LEN,
    ThresholdClassifier,
    build_bilstm_model,
    build_embedding_matrix,
    build_word_index,
    fit_bilstm,
    load_bilstm,
    load_glove,
    predict_probs,
    save_bilstm,
    sequences_to_ids,
    tokenize,
    tune_bilstm,
)


def test_tokenize_lowercases_and_drops_punctuation():
    tokens = tokenize(["Breaking News: World's Fair, 2026!"])
    assert tokens == [["breaking", "news", "world's", "fair", "2026"]]


def test_tokenize_empty_texts():
    assert tokenize(["", None]) == [[], []]


def test_word_index_orders_by_frequency_then_alphabetical():
    seqs = [["b", "a", "a", "b"], ["a"]]
    index = build_word_index(seqs)
    assert index == {"a": 1, "b": 2}


def test_word_index_respects_max_vocab():
    seqs = [["a", "b", "c", "d", "e"]]
    index = build_word_index(seqs, max_vocab=3)
    assert len(index) == 3
    assert set(index.values()) == {1, 2, 3}


def test_word_index_rejects_bad_vocab():
    with pytest.raises(ValueError):
        build_word_index([["a"]], max_vocab=0)


def test_sequences_to_ids_maps_tokens_and_unknowns():
    index = {"a": 1, "b": 2}
    seqs = [["a", "b"], ["b", "zzz", "a"]]
    matrix = sequences_to_ids(seqs, index, max_len=4)
    np.testing.assert_array_equal(
        matrix,
        np.array([[1, 2, 0, 0], [2, 0, 1, 0]], dtype=np.int32),
    )


def test_sequences_to_ids_truncates_first_max_len():
    index = {"a": 1, "b": 2, "c": 3}
    matrix = sequences_to_ids([["a", "b", "c"]], index, max_len=2)
    np.testing.assert_array_equal(matrix, np.array([[1, 2]], dtype=np.int32))


def test_load_glove_parses_file(tmp_path):
    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("a 1.0 2.0\nb 3.0 0.0\n")
    vectors = load_glove(glove_file)
    assert list(vectors.keys()) == ["a", "b"]
    np.testing.assert_array_equal(vectors["a"], np.array([1.0, 2.0], dtype=np.float32))


def test_load_glove_rejects_malformed_lines(tmp_path):
    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("notanumberline\n")
    with pytest.raises(ValueError):
        load_glove(glove_file)


def test_load_glove_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_glove(tmp_path / "missing.txt")


def test_embedding_matrix_covers_known_words_and_pads_unknowns():
    index = {"a": 1, "b": 2, "oov": 3}
    vectors = {"a": np.array([0.5, 0.1], dtype=np.float32)}
    matrix = build_embedding_matrix(index, vectors, embedding_dim=2)
    assert matrix.shape == (4, 2)
    np.testing.assert_array_equal(matrix[0], np.zeros(2))
    np.testing.assert_array_equal(matrix[1], vectors["a"])
    np.testing.assert_array_equal(matrix[2], np.zeros(2))
    np.testing.assert_array_equal(matrix[3], np.zeros(2))


def test_build_bilstm_model_output_shape():
    matrix = np.zeros((4, 2), dtype=np.float32)
    model = build_bilstm_model(vocab_size=4, embedding_matrix=matrix, max_len=5)
    out = model.predict(np.array([1, 2, 0, 0, 0], dtype=np.int32)[None, ...])
    assert out.shape == (1, 1)
    assert not model.get_layer("glove_embedding").trainable


def test_build_bilstm_model_rejects_tiny_vocab():
    with pytest.raises(ValueError):
        build_bilstm_model(vocab_size=1, max_len=5)
    with pytest.raises(ValueError):
        build_bilstm_model(vocab_size=3, dropout=1.0, max_len=5)


def test_tiny_model_fit_and_predict_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.integers(1, 4, size=(12, 5)).astype(np.int32)
    y = np.array([0, 1] * 6)
    val_X = rng.integers(1, 4, size=(4, 5)).astype(np.int32)
    val_y = np.array([0, 1, 0, 1])
    matrix = np.zeros((4, 2), dtype=np.float32)
    model = build_bilstm_model(vocab_size=4, embedding_matrix=matrix, max_len=5)
    history = fit_bilstm(
        model, X, y, val_X, val_y,
        epochs=1, batch_size=4, seed=42,
    )
    assert "loss" in history and len(history["loss"]) == 1
    probs = predict_probs(model, val_X)
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_threshold_classifier_and_metrics_adapter():
    matrix = np.zeros((4, 2), dtype=np.float32)
    model = build_bilstm_model(vocab_size=4, embedding_matrix=matrix, max_len=5)
    probs = predict_probs(model, np.zeros((3, 5), dtype=np.int32))
    expected = (probs >= 0.5).astype(int)
    wrapped = ThresholdClassifier(model)
    np.testing.assert_array_equal(wrapped.predict(np.zeros((3, 5), dtype=np.int32)), expected)
    metrics = evaluate_model(
        wrapped,
        np.zeros((3, 5), dtype=np.int32),
        [0, 1, 0],
    )
    assert metrics["n_samples"] == 3


def test_tune_bilstm_selects_by_validation(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.integers(1, 4, size=(8, 5)).astype(np.int32)
    y = np.array([0, 1] * 4)
    matrix = np.zeros((4, 2), dtype=np.float32)
    best, results = tune_bilstm(
        X, y, X, y, matrix,
        lstm_units=[4], dropouts=[0.2],
        epochs=1, batch_size=4,
        dense_units=4, seed=42,
        max_len=5,
    )
    assert len(results) == 1
    assert results[0]["params"] == {"lstm_units": 4, "dropout": 0.2}
    assert best is not None
    assert hasattr(best, "predict")


def test_tune_bilstm_empty_grid_raises():
    matrix = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        tune_bilstm(np.zeros((2, 5), dtype=np.int32), [0, 1],
                    np.zeros((2, 5), dtype=np.int32), [0, 1], matrix,
                    lstm_units=[], dropouts=[0.3])


def test_save_and_load_roundtrip(tmp_path):
    matrix = np.zeros((4, 2), dtype=np.float32)
    model = build_bilstm_model(vocab_size=4, embedding_matrix=matrix, max_len=5)
    path = tmp_path / "bilstm.keras"
    written = save_bilstm(model, str(path))
    assert Path(written).exists()
    loaded = load_bilstm(written)
    X = np.zeros((2, 5), dtype=np.int32)
    np.testing.assert_array_almost_equal(
        predict_probs(model, X),
        predict_probs(loaded, X),
        decimal=6,
    )


def test_save_rejects_wrong_extension(tmp_path):
    matrix = np.zeros((4, 2), dtype=np.float32)
    model = build_bilstm_model(vocab_size=4, embedding_matrix=matrix, max_len=5)
    with pytest.raises(ValueError):
        save_bilstm(model, str(tmp_path / "model.h5"))


def test_default_config_matches_plan():
    assert DEFAULT_LSTM_UNITS == [64, 128]
    assert DEFAULT_DROPOUTS == [0.3, 0.5]
    assert DEFAULT_MAX_LEN == 300
    assert DEFAULT_BILSTM_PATH == "backend/ml/models/saved/bilstm_model.keras"
    assert list(product(DEFAULT_LSTM_UNITS, DEFAULT_DROPOUTS)) == [
        (64, 0.3), (64, 0.5), (128, 0.3), (128, 0.5)
    ]
