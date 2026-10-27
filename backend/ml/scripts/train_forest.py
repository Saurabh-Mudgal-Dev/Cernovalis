"""Train and optimize the Random Forest classifier on the stratified splits."""

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
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MIN_SAMPLES_SPLIT,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_PATH,
    evaluate_model,
    tune_random_forest,
)
from backend.app.utils.tfidf_vectorizer import (  # noqa: E402
    DEFAULT_VECTORIZER_PATH,
)
from backend.ml.scripts.train_baselines import (  # noqa: E402
    get_vectorizer,
    load_dataset,
    split_docs_labels,
    transform_splits,
)

RF_REPORT_PATH = Path("backend/ml/models/saved/random_forest_report.json")


def parse_depth(value: str) -> int | None:
    """Accept 'none' as an unlimited-depth sentinel for CLI overrides."""
    return None if value.lower() in ("none", "null") else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and optimize the Random Forest classifier."
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
        "--forest-out",
        type=Path,
        default=DEFAULT_RF_PATH,
        help="Output path for the tuned random forest joblib",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=RF_REPORT_PATH,
        help="Output path for the metrics report JSON",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=DEFAULT_RF_N_ESTIMATORS,
        help="Trees per grid configuration",
    )
    parser.add_argument(
        "--max-depths",
        type=str,
        nargs="+",
        default=[str(d) if d is not None else "none" for d in DEFAULT_RF_MAX_DEPTH],
        help="Max tree depths per grid configuration ('none' for unlimited)",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        nargs="+",
        default=DEFAULT_RF_MIN_SAMPLES_SPLIT,
        help="Min split samples per grid configuration",
    )
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    docs, labels_split = split_docs_labels(df)
    vectorizer, fitted = get_vectorizer(
        args.vectorizer_path, docs, force_fit=args.fit_vectorizer
    )
    features = transform_splits(vectorizer, docs)

    max_depths = [parse_depth(d) for d in args.max_depths]
    model, tuning = tune_random_forest(
        features["train"],
        labels_split["train"],
        features["val"],
        labels_split["val"],
        n_estimators=args.n_estimators,
        max_depths=max_depths,
        min_samples_splits=args.min_samples_split,
    )

    report = {
        "random_forest": {
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
    Path(args.forest_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.forest_out)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    best = max(tuning, key=lambda r: r["val_metrics"]["macro_f1"])
    print(f"random_forest: params={best['params']} "
          f"accuracy={report['random_forest']['test_metrics']['accuracy']:.4f} "
          f"macro_f1={report['random_forest']['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
