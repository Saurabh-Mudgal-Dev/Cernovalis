"""Analyze model errors on the held-out test split without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.utils.dataset_split import load_splits  # noqa: E402
from backend.app.utils.ensemble import (  # noqa: E402
    DEFAULT_META_PATH,
    DEFAULT_THRESHOLD,
    get_probabilities,
    soft_vote,
    stacked_matrix,
)
from backend.app.utils.error_analysis import (  # noqa: E402
    analyze_misclassifications,
    shared_error_samples,
    summarize_errors,
)
from backend.app.utils.tfidf_vectorizer import (  # noqa: E402
    DEFAULT_VECTORIZER_PATH,
    TfidfVectorizerWrapper,
)
from backend.ml.scripts.train_baselines import (  # noqa: E402
    load_dataset,
    split_docs_labels,
)
from backend.ml.scripts.train_ensemble import (  # noqa: E402
    BILSTM_PATH,
    TFIDF_MODEL_PATHS,
    _bilstm_sequence_features,
)

DATA_DIR = Path("backend/ml/data/raw")
ENSEMBLE_REPORT_PATH = Path("backend/ml/models/saved/ensemble_report.json")
REPORT_PATH = Path("backend/ml/models/saved/error_analysis.json")


def member_order(available: list[str]) -> list[str]:
    """Use the stored meta-classifier feature order where it is available."""
    if ENSEMBLE_REPORT_PATH.exists():
        with open(ENSEMBLE_REPORT_PATH, "r", encoding="utf-8") as handle:
            coefficients = json.load(handle).get("meta_model_coefficients", {})
        if coefficients:
            return list(coefficients)
    return available


def ensemble_weights() -> dict | None:
    if ENSEMBLE_REPORT_PATH.exists():
        with open(ENSEMBLE_REPORT_PATH, "r", encoding="utf-8") as handle:
            voting = json.load(handle).get("voting", {})
        return voting.get("validation_scored", {}).get("weights")
    return None


def add_model_errors(
    report: dict,
    errors_by_model: dict,
    name: str,
    indices,
    y_true,
    y_pred,
    probabilities_real,
    titles,
    texts,
    text_chars: int,
) -> None:
    errors = analyze_misclassifications(
        indices,
        y_true,
        y_pred,
        probabilities_real,
        titles,
        texts,
        text_chars=text_chars,
    )
    errors_by_model[name] = errors
    report["models"][name] = {
        "summary": summarize_errors(errors, len(y_true)),
        "errors": errors,
    }
    summary = report["models"][name]["summary"]
    print(
        f"{name}: {summary['n_errors']} errors / {summary['n_samples']} "
        f"({summary['error_rate']:.2%}); FP={summary['false_positives']} "
        f"FN={summary['false_negatives']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze misclassified articles on the held-out test split."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing True.csv and Fake.csv")
    parser.add_argument("--vectorizer-path", type=Path, default=DEFAULT_VECTORIZER_PATH,
                        help="Train-split TF-IDF vectorizer")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH,
                        help="Output path for the error-analysis JSON")
    parser.add_argument("--text-chars", type=int, default=500,
                        help="Maximum article-body excerpt length stored per error")
    parser.add_argument("--min-shared-models", type=int, default=2,
                        help="Minimum models that must misclassify an article to list it as shared")
    parser.add_argument("--skip-bilstm", action="store_true",
                        help="Ignore the saved BiLSTM even if it exists")
    args = parser.parse_args()
    if args.text_chars < 0:
        parser.error("--text-chars must be non-negative")
    if args.min_shared_models < 2:
        parser.error("--min-shared-models must be at least 2")

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)
    y_test = np.asarray(labels_split["test"], dtype=int)
    split = load_splits()
    test_indices = np.asarray(split.test_idx, dtype=int)
    if len(test_indices) != len(y_test):
        raise ValueError("test indices and test labels do not align")
    titles = df.iloc[test_indices]["title"].fillna("").astype(str).tolist()
    texts = df.iloc[test_indices]["text"].fillna("").astype(str).tolist()

    report: dict = {
        "split": "test",
        "n_test_samples": int(len(y_test)),
        "models": {},
        "notes": {},
    }
    errors_by_model: dict = {}
    test_probabilities: dict[str, np.ndarray] = {}

    vectorizer = None
    if args.vectorizer_path.exists():
        vectorizer = TfidfVectorizerWrapper.load(args.vectorizer_path)
        X_test = vectorizer.transform(docs["test"])
    else:
        X_test = None

    for name, path in TFIDF_MODEL_PATHS.items():
        if not path.exists():
            report["notes"][name] = f"model not found at {path}"
            print(f"{name}: skipped (model not found)")
            continue
        if X_test is None:
            report["notes"][name] = f"vectorizer not found at {args.vectorizer_path}"
            print(f"{name}: skipped (vectorizer not found)")
            continue
        model = joblib.load(path)
        scores = get_probabilities(model, X_test)
        predictions = np.asarray(model.predict(X_test), dtype=int)
        test_probabilities[name] = scores
        add_model_errors(
            report, errors_by_model, name, test_indices, y_test, predictions,
            scores, titles, texts, args.text_chars,
        )

    if args.skip_bilstm:
        report["notes"]["bilstm"] = "skipped by --skip-bilstm"
    elif not BILSTM_PATH.exists():
        report["notes"]["bilstm"] = f"model not found at {BILSTM_PATH}"
    else:
        try:
            from backend.app.utils.bilstm import load_bilstm, predict_probs

            sequence_features, _, _ = _bilstm_sequence_features(docs, ("train", "test"))
            model = load_bilstm(str(BILSTM_PATH))
            scores = predict_probs(model, sequence_features["test"])
            predictions = (scores >= DEFAULT_THRESHOLD).astype(int)
            test_probabilities["bilstm"] = scores
            add_model_errors(
                report, errors_by_model, "bilstm", test_indices, y_test,
                predictions, scores, titles, texts, args.text_chars,
            )
        except Exception as error:
            report["notes"]["bilstm"] = f"{type(error).__name__}: {error}"
            print(f"bilstm: skipped ({error})")

    order = member_order(list(test_probabilities))
    if not order:
        note = "no trained model probabilities available"
        report["notes"]["soft_vote"] = note
        report["notes"]["meta_classifier"] = note
        missing_members = []
    else:
        missing_members = [name for name in order if name not in test_probabilities]
    if missing_members:
        note = f"missing ensemble members: {missing_members}"
        report["notes"]["soft_vote"] = note
        report["notes"]["meta_classifier"] = note
        print(f"soft_vote/meta_classifier: skipped ({note})")
    elif not order:
        print(f"soft_vote/meta_classifier: skipped ({note})")
    else:
        member_probs = {name: test_probabilities[name] for name in order}
        scores = soft_vote(member_probs, ensemble_weights())
        predictions = (scores >= DEFAULT_THRESHOLD).astype(int)
        add_model_errors(
            report, errors_by_model, "soft_vote", test_indices, y_test,
            predictions, scores, titles, texts, args.text_chars,
        )

        meta_path = Path(DEFAULT_META_PATH)
        if not meta_path.exists():
            report["notes"]["meta_classifier"] = f"model not found at {meta_path}"
        else:
            meta_model = joblib.load(meta_path)
            scores = get_probabilities(meta_model, stacked_matrix(member_probs))
            predictions = (scores >= DEFAULT_THRESHOLD).astype(int)
            add_model_errors(
                report, errors_by_model, "meta_classifier", test_indices,
                y_test, predictions, scores, titles, texts, args.text_chars,
            )

    report["shared_errors"] = shared_error_samples(
        errors_by_model, min_models=args.min_shared_models
    )
    report["n_shared_errors"] = len(report["shared_errors"])
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nshared errors: {report['n_shared_errors']}")
    print(f"report saved to {args.report_out}")


if __name__ == "__main__":
    main()
