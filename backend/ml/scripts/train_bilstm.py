"""Train and tune the GloVe-embedding BiLSTM baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.app.utils.baselines import evaluate_model  # noqa: E402
from backend.app.utils.bilstm import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BILSTM_PATH,
    DEFAULT_DENSE_UNITS,
    DEFAULT_DROPOUTS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_GLOVE_PATH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LSTM_UNITS,
    DEFAULT_MAX_LEN,
    DEFAULT_MAX_VOCAB,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    ThresholdClassifier,
    build_embedding_matrix,
    build_word_index,
    load_glove,
    download_glove,
    save_bilstm,
    sequences_to_ids,
    tokenize,
    tune_bilstm,
)
from backend.ml.scripts.train_baselines import (  # noqa: E402
    load_dataset,
    split_docs_labels,
)

DATA_DIR = Path("backend/ml/data/raw")
REPORT_PATH = Path("backend/ml/models/saved/bilstm_report.json")


def _number_list(text: str):
    return text.replace(",", " ").split()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and tune the GloVe-based BiLSTM baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing True.csv and Fake.csv")
    parser.add_argument("--glove-path", type=Path, default=DEFAULT_GLOVE_PATH,
                        help="Path to glove.6B.100d.txt; downloaded there if missing")
    parser.add_argument("--lstm-units", type=lambda s: [int(v) for v in _number_list(s)],
                        default=DEFAULT_LSTM_UNITS, help="Grid of LSTM units, e.g. '64 128'")
    parser.add_argument("--dropouts", type=lambda s: [float(v) for v in _number_list(s)],
                        default=DEFAULT_DROPOUTS, help="Grid of dropout values, e.g. '0.3 0.5'")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Max epochs per config (early stopping cuts it short)")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                        help="Sequence truncation length")
    parser.add_argument("--max-vocab", type=int, default=DEFAULT_MAX_VOCAB,
                        help="Cap on the learned word-index size")
    parser.add_argument("--recurrent-dropout", type=float, default=0.0,
                        help="Nonzero values disable the fast cuDNN LSTM path")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--trainable-embeddings", action="store_true",
                        help="Fine-tune the GloVe matrix instead of freezing it")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-out", type=Path, default=Path(DEFAULT_BILSTM_PATH))
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)

    token_seqs = {name: tokenize(doc_list) for name, doc_list in docs.items()}
    word_index = build_word_index(token_seqs["train"], max_vocab=args.max_vocab)
    print(f"vocabulary: {len(word_index)} words (max {args.max_vocab})")

    features = {
        name: sequences_to_ids(token_seqs[name], word_index, max_len=args.max_len)
        for name in ("train", "val", "test")
    }

    if not args.glove_path.exists():
        args.glove_path = download_glove(args.glove_path.parent)
    glove_vectors = load_glove(args.glove_path)
    print(f"GloVe vectors: {len(glove_vectors)} entries from {args.glove_path}")

    embedding_matrix = build_embedding_matrix(
        word_index, glove_vectors, embedding_dim=DEFAULT_EMBEDDING_DIM
    )
    covered = int(np.count_nonzero(embedding_matrix.any(axis=1)[1:]))
    print(f"embedding matrix: {embedding_matrix.shape[0]} rows, "
          f"{covered} vocab words covered by GloVe")

    y_train, y_val, y_test = (
        labels_split["train"], labels_split["val"], labels_split["test"]
    )

    best_model, results = tune_bilstm(
        features["train"], y_train,
        features["val"], y_val,
        embedding_matrix,
        lstm_units=args.lstm_units,
        dropouts=args.dropouts,
        recurrent_dropout=args.recurrent_dropout,
        dense_units=DEFAULT_DENSE_UNITS,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        max_len=args.max_len,
        trainable_embeddings=args.trainable_embeddings,
    )

    best_entry = max(results, key=lambda r: r["val_metrics"]["macro_f1"])
    test_metrics = evaluate_model(
        ThresholdClassifier(best_model, DEFAULT_THRESHOLD),
        features["test"], y_test,
    )
    print(f"best params={best_entry['params']} "
          f"test accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_bilstm(best_model, str(args.model_out))

    report = {
        "bilstm": {
            "tuning": results,
            "best_params": best_entry["params"],
            "val_metrics": best_entry["val_metrics"],
            "test_metrics": test_metrics,
        },
        "tokenization": {
            "vocab_size": len(word_index) + 1,
            "max_len": args.max_len,
            "max_vocab": args.max_vocab,
            "glove_path": str(args.glove_path),
            "embedding_dim": DEFAULT_EMBEDDING_DIM,
            "words_covered_by_glove": covered,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "trainable_embeddings": bool(args.trainable_embeddings),
        },
    }
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"model saved to {args.model_out}")
    print(f"report saved to {args.report_out}")


if __name__ == "__main__":
    main()
