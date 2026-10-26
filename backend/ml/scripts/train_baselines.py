"""Train and tune the Logistic Regression and Naive Bayes baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.preprocessor import TextPreprocessor  # noqa: E402
from backend.app.utils.baselines import (  # noqa: E402
    DEFAULT_LOGISTIC_C_VALUES,
    DEFAULT_LOGISTIC_PATH,
    DEFAULT_NB_ALPHAS,
    DEFAULT_NB_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_LOGISTIC_MAX_ITER,
    evaluate_model,
    tune_logistic_regression,
    tune_naive_bayes,
)
from backend.app.utils.dataset_split import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    load_splits,
)
from backend.app.utils.tfidf_vectorizer import (  # noqa: E402
    DEFAULT_VECTORIZER_PATH,
    TfidfVectorizerWrapper,
    combine_title_text,
)

DATA_DIR = Path("backend/ml/data/raw")
REPORT_PATH = Path("backend/ml/models/saved/baseline_report.json")


def load_dataset(data_dir: Path = DATA_DIR):
    """Load and clean the ISOT dataset, sharing the existing load/clean helpers."""
    from backend.ml.scripts.train_features import clean_dataframe, load_data

    df = load_data(data_dir)
    preprocessor = TextPreprocessor(use_spacy=False, remove_stopwords=True)
    df["title_clean"] = df["title"].fillna("").apply(preprocessor.clean)
    df["text_clean"] = df["text"].fillna("").apply(preprocessor.clean)
    return df, preprocessor


def split_docs_labels(df):
    """Slice cleaned titles/texts/labels into train/val/test docs via the split manifest."""
    split = load_splits(DEFAULT_SPLIT_PATH)
    indices = split.subset(range(len(df)))

    title_list = df["title_clean"].tolist()
    text_list = df["text_clean"].tolist()
    labels = df["label"].tolist()

    docs: dict[str, list[str]] = {}
    labels_split: dict[str, list[int]] = {}
    for name, split_indices in zip(("train", "val", "test"), indices):
        docs[name] = combine_title_text(
            [title_list[i] for i in split_indices],
            [text_list[i] for i in split_indices],
        )
        labels_split[name] = [labels[i] for i in split_indices]
    return docs, labels_split


def transform_splits(vectorizer, docs):
    """Transform the per-split document lists into TF-IDF feature matrices."""
    return {name: vectorizer.transform(doc_list) for name, doc_list in docs.items()}


def get_vectorizer(vectorizer_path, docs: dict[str, list[str]], force_fit: bool):
    """Load the train-split TF-IDF vectorizer from disk, or fit and save a fresh one."""
    if not force_fit and Path(vectorizer_path).exists():
        return TfidfVectorizerWrapper.load(vectorizer_path), False
    vectorizer = TfidfVectorizerWrapper()
    vectorizer.fit(docs["train"])
    vectorizer.save(vectorizer_path)
    return vectorizer, True


def build_features(df, vectorizer_path):
    """Fit the TF-IDF vectorizer on the train split and transform all splits."""
    docs, labels_split = split_docs_labels(df)

    vectorizer = TfidfVectorizerWrapper()
    vectorizer.fit(docs["train"])
    vectorizer.save(vectorizer_path)

    features = transform_splits(vectorizer, docs)
    return features, labels_split, vectorizer


def tune_and_report(features, labels_split):
    """Run both tuning searches and score the winners on the held-out test split."""
    X_train, X_val, X_test = features["train"], features["val"], features["test"]
    y_train, y_val, y_test = labels_split["train"], labels_split["val"], labels_split["test"]

    logistic_model, logistic_tuning = tune_logistic_regression(
        X_train, y_train, X_val, y_val,
        C_values=DEFAULT_LOGISTIC_C_VALUES,
        max_iter=DEFAULT_LOGISTIC_MAX_ITER,
        random_state=DEFAULT_RANDOM_STATE,
    )
    nb_model, nb_tuning = tune_naive_bayes(
        X_train, y_train, X_val, y_val,
        alphas=DEFAULT_NB_ALPHAS,
    )

    report = {
        "logistic_regression": {
            "tuning": logistic_tuning,
            "val_metrics": evaluate_model(logistic_model, X_val, y_val),
            "test_metrics": evaluate_model(logistic_model, X_test, y_test),
        },
        "naive_bayes": {
            "tuning": nb_tuning,
            "val_metrics": evaluate_model(nb_model, X_val, y_val),
            "test_metrics": evaluate_model(nb_model, X_test, y_test),
        },
    }
    return logistic_model, nb_model, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and tune the Logistic Regression and Naive Bayes baselines."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--vectorizer-out",
        type=Path,
        default=DEFAULT_VECTORIZER_PATH,
        help="Output path for the train-split TF-IDF vectorizer joblib",
    )
    parser.add_argument(
        "--logistic-out",
        type=Path,
        default=DEFAULT_LOGISTIC_PATH,
        help="Output path for the tuned logistic regression joblib",
    )
    parser.add_argument(
        "--nb-out",
        type=Path,
        default=DEFAULT_NB_PATH,
        help="Output path for the tuned naive bayes joblib",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPORT_PATH,
        help="Output path for the metrics report JSON",
    )
    args = parser.parse_args()

    df, _ = load_dataset(args.data_dir)
    features, labels_split, vectorizer = build_features(df, args.vectorizer_out)
    logistic_model, nb_model, report = tune_and_report(features, labels_split)

    Path(args.logistic_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, args.logistic_out)
    joblib.dump(nb_model, args.nb_out)
    report["vectorizer"] = {
        "path": str(args.vectorizer_out),
        "n_features": vectorizer.n_features,
        "train_docs": int(features["train"].shape[0]),
    }
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for name, entry in report.items():
        if not isinstance(entry, dict) or "test_metrics" not in entry:
            continue
        best = max(entry["tuning"], key=lambda r: r["val_metrics"]["macro_f1"])
        print(f"{name}: params={best['params']} accuracy={entry['test_metrics']['accuracy']:.4f} "
              f"macro_f1={entry['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
