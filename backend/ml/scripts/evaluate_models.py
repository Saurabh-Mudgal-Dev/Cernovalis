"""Generate confusion matrices, ROC-AUC, precision-recall curves and feature importance."""

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

from backend.app.utils.baselines import evaluate_model  # noqa: E402
from backend.app.utils.ensemble import (  # noqa: E402
    DEFAULT_META_PATH,
    DEFAULT_THRESHOLD,
    evaluate_probabilities,
    get_probabilities,
    soft_vote,
    stacked_matrix,
)
from backend.app.utils.evaluation import (  # noqa: E402
    DEFAULT_TOP_K,
    feature_importance,
    plot_confusion,
    plot_feature_importance,
    plot_pr,
    plot_roc,
    pr_points,
    resample_curve,
    roc_points,
    row_normalized_confusion,
)
from backend.app.utils.tfidf_vectorizer import DEFAULT_VECTORIZER_PATH  # noqa: E402
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
REPORT_PATH = Path("backend/ml/models/saved/evaluation_report.json")
FIGURES_DIR = Path("backend/ml/reports/figures")
BILSTM_KEY = "bilstm"


def compact_roc(roc: dict) -> dict:
    fpr, tpr = resample_curve(roc["fpr"], roc["tpr"])
    return {"fpr": fpr, "tpr": tpr}


def compact_pr(pr: dict) -> dict:
    recall, precision = resample_curve(
        np.asarray(pr["recall"])[::-1], np.asarray(pr["precision"])[::-1]
    )
    return {"precision": precision, "recall": recall}


def score_entry(
    y_true,
    y_pred,
    scores,
    threshold_metrics: dict,
    model=None,
    feature_names=None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[dict, dict, dict]:
    """Build a model's report entry plus its full ROC and PR curves for the overlay plots."""
    roc = roc_points(y_true, scores)
    pr = pr_points(y_true, scores)
    entry: dict = {
        "threshold_metrics": threshold_metrics,
        "confusion_matrix_normalized": row_normalized_confusion(y_true, y_pred),
        "roc_auc": roc["roc_auc"],
        "average_precision": pr["average_precision"],
        "curves": {"roc": compact_roc(roc), "pr": compact_pr(pr)},
    }
    if model is not None:
        importance = feature_importance(model, feature_names, top_k)
        if importance:
            entry["feature_importance"] = importance
    return entry, roc, pr


def member_order(available: list[str]) -> list[str]:
    """Prefer the column order the meta-classifier was trained with, else the available members."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices, ROC-AUC, precision-recall curves "
                    "and feature importance for the trained models."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing True.csv and Fake.csv")
    parser.add_argument("--vectorizer-path", type=Path, default=DEFAULT_VECTORIZER_PATH,
                        help="Train-split TF-IDF vectorizer saved by train_baselines.py")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH,
                        help="Output path for the evaluation report JSON")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Directory for the rendered PNG figures")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of top features kept per model")
    parser.add_argument("--skip-bilstm", action="store_true",
                        help="Ignore the saved BiLSTM even if it exists")
    parser.add_argument("--no-plots", action="store_true",
                        help="Compute the report without rendering figures")
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)
    y_test = labels_split["test"]

    if args.vectorizer_path.exists():
        from backend.app.utils.tfidf_vectorizer import TfidfVectorizerWrapper

        vectorizer = TfidfVectorizerWrapper.load(args.vectorizer_path)
        X_test = vectorizer.transform(docs["test"])
        feature_names = vectorizer.feature_names_
    else:
        X_test, feature_names = None, None

    report: dict = {"split": "test", "n_test_samples": int(len(y_test)), "models": {}}
    notes: dict = {}
    test_probs: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    roc_full: dict[str, dict] = {}
    pr_full: dict[str, dict] = {}

    def record(name, entry, roc, pr, y_pred):
        report["models"][name] = entry
        predictions[name] = y_pred
        roc_full[name] = roc
        pr_full[name] = pr

    def announce(name, entry):
        print(
            f"{name}: roc_auc={entry['roc_auc']:.4f} "
            f"avg_precision={entry['average_precision']:.4f} "
            f"macro_f1={entry['threshold_metrics']['macro_f1']:.4f}"
        )

    for name, path in TFIDF_MODEL_PATHS.items():
        if not path.exists():
            notes[name] = f"skipped: model not found at {path}"
            print(f"{name}: skipped ({path} not found)")
            continue
        if X_test is None:
            notes[name] = f"skipped: vectorizer not found at {args.vectorizer_path}"
            print(f"{name}: skipped (vectorizer not found)")
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        scores = get_probabilities(model, X_test)
        test_probs[name] = scores
        entry, roc, pr = score_entry(
            y_test, y_pred, scores, evaluate_model(model, X_test, y_test),
            model=model, feature_names=feature_names, top_k=args.top_k,
        )
        record(name, entry, roc, pr, y_pred)
        announce(name, entry)

    if args.skip_bilstm:
        notes[BILSTM_KEY] = "skipped by --skip-bilstm"
    elif not BILSTM_PATH.exists():
        notes[BILSTM_KEY] = f"skipped: model not found at {BILSTM_PATH}"
    elif X_test is None:
        notes[BILSTM_KEY] = f"skipped: vectorizer not found at {args.vectorizer_path}"
    else:
        try:
            from backend.app.utils.bilstm import load_bilstm, predict_probs

            sequence_features, _, _ = _bilstm_sequence_features(docs, ("train", "test"))
            scores = predict_probs(
                load_bilstm(str(BILSTM_PATH)), sequence_features["test"]
            )
            y_pred = (scores >= DEFAULT_THRESHOLD).astype(int)
            test_probs[BILSTM_KEY] = scores
            entry, roc, pr = score_entry(
                y_test, y_pred, scores, evaluate_probabilities(y_test, scores),
            )
            record(BILSTM_KEY, entry, roc, pr, y_pred)
            announce(BILSTM_KEY, entry)
        except Exception as error:
            notes[BILSTM_KEY] = f"skipped: {type(error).__name__}: {error}"
            print(f"{BILSTM_KEY}: skipped ({error})")

    order = member_order(list(test_probs))
    missing_members = [name for name in order if name not in test_probs]
    if missing_members:
        skip = f"skipped: missing members {missing_members}"
        notes["soft_vote"] = skip
        notes["meta_classifier"] = skip
        print(f"soft_vote: skipped (missing {missing_members})")
        print(f"meta_classifier: skipped (missing {missing_members})")
    else:
        member_probs = {name: test_probs[name] for name in order}
        voted = soft_vote(member_probs, ensemble_weights())
        y_pred = (voted >= DEFAULT_THRESHOLD).astype(int)
        entry, roc, pr = score_entry(
            y_test, y_pred, voted, evaluate_probabilities(y_test, voted),
        )
        record("soft_vote", entry, roc, pr, y_pred)
        announce("soft_vote", entry)

        meta_path = Path(DEFAULT_META_PATH)
        if not meta_path.exists():
            notes["meta_classifier"] = f"skipped: model not found at {meta_path}"
            print(f"meta_classifier: skipped ({meta_path} not found)")
        else:
            meta_model = joblib.load(meta_path)
            stacked = stacked_matrix(member_probs)
            meta_scores = get_probabilities(meta_model, stacked)
            y_pred = (meta_scores >= DEFAULT_THRESHOLD).astype(int)
            entry, roc, pr = score_entry(
                y_test, y_pred, meta_scores,
                evaluate_model(meta_model, stacked, y_test),
                model=meta_model, feature_names=order, top_k=args.top_k,
            )
            record("meta_classifier", entry, roc, pr, y_pred)
            announce("meta_classifier", entry)

    report["notes"] = notes
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nreport saved to {args.report_out}")

    if args.no_plots:
        return
    write_figures(args.figures_dir, y_test, predictions, roc_full, pr_full, report)


def write_figures(figures_dir, y_test, predictions, roc_full, pr_full, report) -> None:
    for name, y_pred in predictions.items():
        entry = report["models"][name]
        plot_confusion(
            y_test, y_pred, figures_dir / f"{name}_confusion.png",
            title=f"Confusion matrix — {name}",
        )
        plot_confusion(
            y_test, y_pred, figures_dir / f"{name}_confusion_norm.png",
            title=f"Confusion matrix (row-normalized) — {name}", normalize=True,
        )
        if entry.get("feature_importance"):
            plot_feature_importance(
                entry["feature_importance"],
                figures_dir / f"{name}_feature_importance.png",
                title=f"Feature importance — {name}",
            )
    if roc_full:
        plot_roc(roc_full, figures_dir / "roc_curves.png")
    if pr_full:
        plot_pr(pr_full, figures_dir / "precision_recall_curves.png")
    print(f"figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
