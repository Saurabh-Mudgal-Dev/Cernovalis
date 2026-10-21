"""Train and persist the TF-IDF vectorizer on cleaned article titles + content.

Run from the repository root:
    python backend/ml/scripts/train_tfidf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Resolve project root so the package imports work regardless of CWD.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.preprocessor import TextPreprocessor  # noqa: E402
from backend.app.utils.tfidf_vectorizer import (  # noqa: E402
    TfidfVectorizerWrapper,
    combine_title_text,
    split_train_test,
)

DATA_DIR = Path("backend/ml/data/raw")
VECTORIZER_OUT = Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib")
FEATURES_OUT = Path("backend/ml/models/saved/tfidf_features.npz")


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and label the True.csv / Fake.csv ISOT dataset."""
    true_df = pd.read_csv(data_dir / "True.csv")
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = true_df.assign(label=1, class_name="True")
    fake_df = fake_df.assign(label=0, class_name="Fake")
    df = (
        pd.concat([true_df, fake_df], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return df


def clean_dataframe(
    df: pd.DataFrame,
    use_spacy: bool = False,
) -> pd.DataFrame:
    """Apply the shared preprocessing pipeline to title and text columns."""
    preprocessor = TextPreprocessor(use_spacy=use_spacy, remove_stopwords=True)
    df = df.copy()
    df["title_clean"] = df["title"].fillna("").apply(preprocessor.clean)
    df["text_clean"] = df["text"].fillna("").apply(preprocessor.clean)
    df["content_clean"] = (
        df["title_clean"].fillna("") + " " + df["text_clean"].fillna("")
    ).str.strip()
    return df


def train_vectorizer(
    df: pd.DataFrame,
    vectorizer_path: Path = VECTORIZER_OUT,
    features_path: Path | None = FEATURES_OUT,
    use_spacy: bool = False,
) -> TfidfVectorizerWrapper:
    """Clean, split, fit, and persist the TF-IDF vectorizer.

    Returns the fitted :class:`TfidfVectorizerWrapper`.
    """
    df = clean_dataframe(df, use_spacy=use_spacy)

    titles = df["title_clean"].tolist()
    texts = df["text_clean"].tolist()
    labels = df["label"].tolist()

    (
        train_titles,
        _test_titles,
        train_texts,
        _test_texts,
        train_labels,
        _test_labels,
    ) = split_train_test(titles, texts, labels, test_size=0.30, random_state=42)

    train_docs = combine_title_text(train_titles, train_texts)

    vectorizer = TfidfVectorizerWrapper()
    vectorizer.fit(train_docs)

    vectorizer.save(vectorizer_path)

    if features_path is not None:
        features_path.parent.mkdir(parents=True, exist_ok=True)
        X_train = vectorizer.transform(train_docs)
        try:
            
            from scipy.sparse import save_npz

            save_npz(str(features_path), X_train)
        except ImportError:  # pragma: no cover - scipy always present with sklearn

            import numpy as np

            np.save(str(features_path), X_train.toarray())

    print(
        f"TF-IDF vectorizer fitted on {len(train_docs)} training documents "
        f"with {vectorizer.n_features} features. Saved to {vectorizer_path}"
    )
    return vectorizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist the TF-IDF vectorizer."
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
        default=VECTORIZER_OUT,
        help="Output path for the fitted vectorizer joblib",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=FEATURES_OUT,
        help="Output path for the training feature matrix (npz)",
    )
    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy lemmatization during cleaning (requires en_core_web_sm)",
    )
    args = parser.parse_args()

    df = load_data(args.data_dir)
    train_vectorizer(
        df,
        vectorizer_path=args.vectorizer_out,
        features_path=args.features_out,
        use_spacy=args.use_spacy,
    )


if __name__ == "__main__":
    main()