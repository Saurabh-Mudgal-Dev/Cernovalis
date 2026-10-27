"""Soft-voting ensemble evaluation and meta-classifier stacking on the trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.app.utils.ensemble import (  # noqa: E402
    DEFAULT_META_C_VALUES,
    DEFAULT_META_PATH,
    evaluate_probabilities,
    get_probabilities,
    soft_vote,
    save_meta,
    stacked_matrix,
    tune_meta_classifier,
    weights_from_scores,
)
from backend.ml.scripts.train_baselines import load_dataset, split_docs_labels  # noqa: E402

DATA_DIR = Path("backend/ml/data/raw")
REPORT_PATH = Path("backend/ml/models/saved/ensemble_report.json")

TFIDF_MODEL_PATHS = {
    "logistic_regression": Path("backend/ml/models/saved/logistic_regression.joblib"),
    "naive_bayes": Path("backend/ml/models/saved/naive_bayes.joblib"),
    "random_forest": Path("backend/ml/models/saved/random_forest.joblib"),
    "xgboost": Path("backend/ml/models/saved/xgboost_model.joblib"),
}
BILSTM_PATH = Path("backend/ml/models/saved/bilstm_model.keras")
BILSTM_REPORT_PATH = Path("backend/ml/models/saved/bilstm_report.json")


def _load_tfidf_features(vectorizer_path: Path):
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {vectorizer_path}; run "
            "backend/ml/scripts/train_baselines.py first"
        )
    from backend.app.utils.tfidf_vectorizer import TfidfVectorizerWrapper

    vectorizer = TfidfVectorizerWrapper.load(vectorizer_path)
    return vectorizer


def _bilstm_sequence_features(docs: dict[str, list[str]], splits: Sequence[str]):
    """Rebuild the BiLSTM tokenization deterministically from the shared pipeline."""
    from backend.app.utils.bilstm import DEFAULT_EMBEDDING_DIM, build_word_index, sequences_to_ids, tokenize

    max_len, max_vocab = 300, 50000
    if BILSTM_REPORT_PATH.exists():
        with open(BILSTM_REPORT_PATH, "r", encoding="utf-8") as handle:
            tokenization = json.load(handle).get("tokenization", {})
        max_len = int(tokenization.get("max_len", max_len))
        max_vocab = int(tokenization.get("max_vocab", max_vocab))

    token_seqs = {name: tokenize(docs[name]) for name in splits}
    word_index = build_word_index(token_seqs["train"], max_vocab=max_vocab)
    features = {
        name: sequences_to_ids(token_seqs[name], word_index, max_len=max_len)
        for name in splits
    }
    return features, len(word_index) + 1, max_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the soft-voting ensemble and train the stacked meta-classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--vectorizer-path", type=Path,
        default=Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib"),
        help="Train-split TF-IDF vectorizer saved by train_baselines.py",
    )
    parser.add_argument("--skip-bilstm", action="store_true",
                        help="Ignore the saved BiLSTM even if it exists")
    parser.add_argument("--model-out", type=Path, default=Path(DEFAULT_META_PATH))
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)
    y_val, y_test = labels_split["val"], labels_split["test"]

    vectorizer = _load_tfidf_features(args.vectorizer_path)
    tfidf_features = {
        "val": vectorizer.transform(docs["val"]),
        "test": vectorizer.transform(docs["test"]),
    }

    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    model_notes: dict[str, str] = {}

    for name, path in TFIDF_MODEL_PATHS.items():
        if not path.exists():
            model_notes[name] = f"skipped: model not found at {path}"
            print(f"{name}: skipped ({path} not found)")
            continue
        model = __import__("joblib").load(path)
        val_probs[name] = get_probabilities(model, tfidf_features["val"])
        test_probs[name] = get_probabilities(model, tfidf_features["test"])
        print(f"{name}: probabilities collected")

    if not args.skip_bilstm and BILSTM_PATH.exists():
        try:
            from backend.app.utils.bilstm import load_bilstm, predict_probs

            sequence_features, n_vocab, max_len = _bilstm_sequence_features(
                docs, ("train", "val", "test")
            )
            bilstm_model = load_bilstm(str(BILSTM_PATH))
            val_probs["bilstm"] = predict_probs(bilstm_model, sequence_features["val"])
            test_probs["bilstm"] = predict_probs(bilstm_model, sequence_features["test"])
            print(f"bilstm: probabilities collected (vocab {n_vocab}, max_len {max_len})")
        except Exception as error:  # skip a mismatched BiLSTM rather than fail the run
            model_notes["bilstm"] = f"skipped: {type(error).__name__}: {error}"
            print(f"bilstm: skipped ({error})")
    elif args.skip_bilstm:
        model_notes["bilstm"] = "skipped by --skip-bilstm"
    else:
        model_notes["bilstm"] = f"skipped: model not found at {BILSTM_PATH}"
        print(f"bilstm: skipped ({BILSTM_PATH} not found)")

    if len(val_probs) < 2:
        raise SystemExit(
            "need at least two trained models; run the train scripts first"
        )

    val_scores = {
        name: evaluate_probabilities(y_val, probs)["macro_f1"]
        for name, probs in val_probs.items()
    }
    print("\nper-model validation macro F1:")
    for name, score in val_scores.items():
        print(f"  {name}: {score:.4f}")

    tuned_weights = weights_from_scores(val_scores)
    print(f"validation-scored weights: {tuned_weights}")

    report = {
        "members": {name: {"val_macro_f1": val_scores[name]} for name in val_probs},
        "notes": model_notes,
    }

    voting = {}
    vote_configs = {
        "equal_weights": None,
        "validation_scored": tuned_weights,
    }
    print("\nsoft voting on the test split:")
    for config_name, weights in vote_configs.items():
        effective = weights or {name: 1.0 for name in val_probs}
        val_metrics = evaluate_probabilities(y_val, soft_vote(val_probs, effective))
        test_metrics = evaluate_probabilities(y_test, soft_vote(test_probs, effective))
        voting[config_name] = {
            "weights": dict(effective),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        print(f"  {config_name}: test accuracy={test_metrics['accuracy']:.4f} "
              f"macro_f1={test_metrics['macro_f1']:.4f}")
    report["voting"] = voting

    print("\nmeta-classifier stacking:")
    meta_model, meta_tuning = tune_meta_classifier(
        stacked_matrix(val_probs), y_val, C_values=DEFAULT_META_C_VALUES,
    )
    meta_val_probs = meta_model.predict_proba(stacked_matrix(val_probs))[:, -1]
    meta_test_probs = meta_model.predict_proba(stacked_matrix(test_probs))[:, -1]
    meta_report = {
        "tuning": meta_tuning,
        "val_metrics": evaluate_probabilities(y_val, meta_val_probs),
        "test_metrics": evaluate_probabilities(y_test, meta_test_probs),
    }
    best_c = max(meta_tuning, key=lambda r: r["cv_macro_f1"])
    print(f"  best C={best_c['params']['C']} cv_macro_f1={best_c['cv_macro_f1']:.4f}")
    print(f"  test accuracy={meta_report['test_metrics']['accuracy']:.4f} "
          f"macro_f1={meta_report['test_metrics']['macro_f1']:.4f}")
    report["meta_classifier"] = meta_report

    model_importance = meta_model.coef_.ravel().tolist()
    report["meta_model_coefficients"] = {
        name: float(coef) for name, coef in zip(val_probs, model_importance)
    }

    best_meta = meta_report["test_metrics"]["macro_f1"]
    final_name, final_entry = max(
        voting.items(), key=lambda kv: kv[1]["test_metrics"]["macro_f1"]
    )
    if best_meta > final_entry["test_metrics"]["macro_f1"]:
        print("\nfinal: stacked meta-classifier wins by macro F1")
    else:
        print(f"\nfinal: soft voting ({final_name}) wins by macro F1")
    save_meta(meta_model, str(args.model_out))
    print(f"meta-classifier saved to {args.model_out}")

    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"report saved to {args.report_out}")


if __name__ == "__main__":
    main()
