"""Train and optimize the XGBoost classifier on the stratified splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.utils.baselines import (  # noqa: E402
    DEFAULT_XGB_LEARNING_RATE,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_N_ESTIMATORS,
    DEFAULT_XGB_N_JOBS,
    DEFAULT_XGB_PATH,
    evaluate_model,
    tune_xgboost,
)
from backend.app.utils.tfidf_vectorizer import DEFAULT_VECTORIZER_PATH  # noqa: E402
from backend.ml.scripts.train_baselines import (  # noqa: E402
    get_vectorizer,
    load_dataset,
    split_docs_labels,
    transform_splits,
)

XGB_REPORT_PATH = Path("backend/ml/models/saved/xgboost_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and optimize the XGBoost classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("backend/ml/data/raw"),
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=Path,
        default=DEFAULT_VECTORIZER_PATH,
        help="Path of the train-split TF-IDF vectorizer joblib",
    )
    parser.add_argument(
        "--fit-vectorizer",
        action="store_true",
        help="Refit the TF-IDF vectorizer on the train split even if one exists",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_XGB_PATH,
        help="Output path for the tuned XGBoost joblib",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=XGB_REPORT_PATH,
        help="Output path for the metrics report JSON",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=DEFAULT_XGB_N_ESTIMATORS,
        help="Boosting rounds per grid configuration",
    )
    parser.add_argument(
        "--max-depths",
        type=int,
        nargs="+",
        default=DEFAULT_XGB_MAX_DEPTH,
        help="Max tree depths per grid configuration",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=DEFAULT_XGB_LEARNING_RATE,
        help="Learning rates per grid configuration",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_XGB_N_JOBS,
        help="Parallel workers per fit (-1 for all cores)",
    )
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)
    vectorizer, fitted = get_vectorizer(
        args.vectorizer_path, docs, force_fit=args.fit_vectorizer
    )
    features = transform_splits(vectorizer, docs)

    model, tuning = tune_xgboost(
        features["train"],
        labels_split["train"],
        features["val"],
        labels_split["val"],
        n_estimators=args.n_estimators,
        max_depths=args.max_depths,
        learning_rates=args.learning_rates,
        n_jobs=args.n_jobs,
        random_state=42,
    )

    report = {
        "xgboost": {
            "tuning": tuning,
            "val_metrics": evaluate_model(model, features["val"], labels_split["val"]),
            "test_metrics": evaluate_model(model, features["test"], labels_split["test"]),
        },
        "vectorizer": {
            "path": str(args.vectorizer_path),
            "fitted_now": fitted,
            "n_features": vectorizer.n_features,
            "train_docs": int(features["train"].shape[0]),
        },
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    best = max(tuning, key=lambda r: r["val_metrics"]["macro_f1"])
    print(f"xgboost: params={best['params']} "
          f"accuracy={report['xgboost']['test_metrics']['accuracy']:.4f} "
          f"macro_f1={report['xgboost']['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
