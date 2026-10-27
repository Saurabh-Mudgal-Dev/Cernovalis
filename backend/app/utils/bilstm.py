"""Tokenization, GloVe embedding loading and BiLSTM model plumbing for the deep-learning track."""

from __future__ import annotations

import re
import urllib.request
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

DEFAULT_EMBEDDINGS_DIR = Path("backend/ml/models/embeddings")
DEFAULT_GLOVE_PATH = DEFAULT_EMBEDDINGS_DIR / "glove.6B.100d.txt"
GLOVE_DOWNLOAD_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_ZIP_MEMBER = "glove.6B.100d.txt"

DEFAULT_MAX_LEN = 300
DEFAULT_MAX_VOCAB = 50000
DEFAULT_EMBEDDING_DIM = 100

DEFAULT_LSTM_UNITS = [64, 128]
DEFAULT_DROPOUTS = [0.3, 0.5]
DEFAULT_DENSE_UNITS = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 3
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5

DEFAULT_BILSTM_PATH = "backend/ml/models/saved/bilstm_model.keras"

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(texts: Iterable[str]) -> List[List[str]]:
    """Lowercase and split texts into word tokens (letters, digits, apostrophes)."""
    return [
        TOKEN_PATTERN.findall(text.lower()) if isinstance(text, str) else []
        for text in texts
    ]


def build_word_index(
    token_seqs: Sequence[Sequence[str]],
    max_vocab: int = DEFAULT_MAX_VOCAB,
) -> Dict[str, int]:
    """Map the `max_vocab` most frequent tokens to 1-based ids.

    Ties in frequency are broken alphabetically so the index is deterministic.
    Id 0 is reserved for padding / unknown tokens.
    """
    if max_vocab < 1:
        raise ValueError("max_vocab must be at least 1")
    counts = Counter(token for seq in token_seqs for token in seq)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_vocab]
    return {word: index for index, (word, _) in enumerate(ranked, start=1)}


def sequences_to_ids(
    token_seqs: Sequence[Sequence[str]],
    word_index: Dict[str, int],
    max_len: int = DEFAULT_MAX_LEN,
) -> np.ndarray:
    """Convert token sequences into a padded id matrix (truncating to the first `max_len` tokens)."""
    if max_len < 1:
        raise ValueError("max_len must be at least 1")
    matrix = np.zeros((len(token_seqs), max_len), dtype=np.int32)
    for row, seq in enumerate(token_seqs):
        ids = [word_index.get(token, 0) for token in seq[:max_len]]
        matrix[row, : len(ids)] = ids
    return matrix


def load_glove(path: Path | str = DEFAULT_GLOVE_PATH) -> Dict[str, np.ndarray]:
    """Parse a GloVe text file into a {word: vector} table."""
    glove_path = Path(path)
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe embeddings not found at {glove_path}. "
            "Run with --glove-path pointing at glove.6B.100d.txt to download it."
        )
    vectors: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                raise ValueError(f"malformed GloVe line in {glove_path}: {line[:80]!r}")
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def download_glove(save_dir: Path | str = DEFAULT_EMBEDDINGS_DIR) -> Path:
    """Download the GloVe zip and extract glove.6B.100d.txt into `save_dir`."""
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / GLOVE_ZIP_MEMBER
    if target_file.exists():
        return target_file
    print(f"downloading GloVe embeddings from {GLOVE_DOWNLOAD_URL} ...")
    zip_path, _ = urllib.request.urlretrieve(GLOVE_DOWNLOAD_URL)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(GLOVE_ZIP_MEMBER, path=str(target_dir))
    if not target_file.exists():
        raise RuntimeError(f"GloVe extraction failed: {target_file} not found")
    return target_file


def build_embedding_matrix(
    word_index: Dict[str, int],
    glove_vectors: Dict[str, np.ndarray],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> np.ndarray:
    """Stack GloVe vectors into a (len(word_index) + 1) x dim matrix; row 0 is all zeros."""
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1")
    matrix = np.zeros((len(word_index) + 1, embedding_dim), dtype=np.float32)
    for word, index in word_index.items():
        vector = glove_vectors.get(word)
        if vector is not None and vector.shape == (embedding_dim,):
            matrix[index] = vector
    return matrix


def build_bilstm_model(
    vocab_size: int,
    embedding_matrix: np.ndarray | None = None,
    lstm_units: int = 128,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    dense_units: int = DEFAULT_DENSE_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    trainable_embeddings: bool = False,
    max_len: int = DEFAULT_MAX_LEN,
):
    """Build the BiLSTM classifier: Embedding -> SpatialDropout -> BiLSTM -> pooled -> Dense sigmoid."""
    import keras

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 (pad + one word)")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    initializer = (
        keras.initializers.Constant(embedding_matrix)
        if embedding_matrix is not None
        else "uniform"
    )
    inputs = keras.Input(shape=(max_len,), name="tokens")
    x = keras.layers.Embedding(
        vocab_size,
        embedding_matrix.shape[1] if embedding_matrix is not None else DEFAULT_EMBEDDING_DIM,
        embeddings_initializer=initializer,
        trainable=trainable_embeddings,
        name="glove_embedding",
    )(inputs)
    x = keras.layers.SpatialDropout1D(dropout)(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=recurrent_dropout)
    )(x)
    x = keras.layers.Concatenate()(
        [
            keras.layers.GlobalMaxPooling1D()(x),
            keras.layers.GlobalAveragePooling1D()(x),
        ]
    )
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="bilstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_bilstm(
    model,
    X_train,
    y_train: Sequence[int],
    X_val,
    y_val: Sequence[int],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> dict:
    """Train with early stopping on validation loss and restore the best weights."""
    import keras

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    keras.utils.set_random_seed(seed)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        np.asarray(X_train),
        np.asarray(y_train, dtype=np.float32),
        validation_data=(np.asarray(X_val), np.asarray(y_val, dtype=np.float32)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def predict_probs(model, X) -> np.ndarray:
    """Return the model's sigmoid output per row, squeezed to a 1-D array."""
    return np.asarray(model.predict(X, verbose=0)).ravel()


class ThresholdClassifier:
    """Adapter so sklearn-style metric helpers can consume a Keras probability model."""

    def __init__(self, model, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.model = model
        self.threshold = threshold

    def predict(self, X) -> np.ndarray:
        return (predict_probs(self.model, X) >= self.threshold).astype(int)


def tune_bilstm(
    X_train,
    y_train: Sequence[int],
    X_val,
    y_val: Sequence[int],
    embedding_matrix: np.ndarray,
    lstm_units: Sequence[int] = DEFAULT_LSTM_UNITS,
    dropouts: Sequence[float] = DEFAULT_DROPOUTS,
    recurrent_dropout: float = 0.0,
    dense_units: int = DEFAULT_DENSE_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
    max_len: int = DEFAULT_MAX_LEN,
    trainable_embeddings: bool = False,
) -> tuple[object, List[Dict[str, object]]]:
    """Fit every grid combination, pick the best by validation macro F1."""
    from backend.app.utils.baselines import _tune

    if not lstm_units or not dropouts:
        raise ValueError("lstm_units and dropouts lists must not be empty")

    models = []
    for index, (lstm_value, dropout_value) in enumerate(product(lstm_units, dropouts)):
        print(f"fitting lstm_units={lstm_value} dropout={dropout_value} ...")
        model = build_bilstm_model(
            vocab_size=embedding_matrix.shape[0],
            embedding_matrix=embedding_matrix,
            lstm_units=lstm_value,
            dropout=dropout_value,
            recurrent_dropout=recurrent_dropout,
            dense_units=dense_units,
            learning_rate=learning_rate,
            trainable_embeddings=trainable_embeddings,
            max_len=max_len,
        )
        history = fit_bilstm(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed + index,
        )
        epochs_run = len(history["val_loss"])
        print(f"  trained for {epochs_run} epochs (early stopping patience {patience})")
        models.append(ThresholdClassifier(model, threshold))

    names = [
        {"lstm_units": lstm_value, "dropout": dropout_value}
        for lstm_value, dropout_value in product(lstm_units, dropouts)
    ]
    best, results = _tune(models, names, X_val, y_val)
    return best.model, results


def save_bilstm(model, path: str = DEFAULT_BILSTM_PATH) -> str:
    """Persist a fitted model in the native .keras format."""
    import keras

    if not str(path).endswith(".keras"):
        raise ValueError("BiLSTM model path must end with .keras")
    model.save(path)
    return path


def load_bilstm(path: str = DEFAULT_BILSTM_PATH):
    """Load a model previously written by save_bilstm."""
    import keras

    model = keras.saving.load_model(path)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    return model
