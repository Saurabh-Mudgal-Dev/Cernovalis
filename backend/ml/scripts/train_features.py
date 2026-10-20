"""Train and persist the unified feature pipeline (TF-IDF + stylometric features).

Run from the repository root:

    python backend/ml/scripts/train_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.preprocessor import TextPreprocessor  # noqa: E402
from backend.app.utils.feature_pipeline import (  # noqa: E402
    DEFAULT_PIPELINE_PATH,
    FeaturePipeline,
)

DATA_DIR = Path("backend/ml/data/raw")
PIPELINE_OUT = DEFAULT_PIPELINE_PATH
MATRIX_OUT = Path("backend/ml/models/saved/features.npz")


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


def clean_dataframe(df: pd.DataFrame, use_spacy: bool = False) -> pd.DataFrame:
    """Apply the shared preprocessing pipeline to title and text columns."""
    preprocessor = TextPreprocessor(use_spacy=use_spacy, remove_stopwords=True)
    df = df.copy()
    df["title_clean"] = df["title"].fillna("").apply(preprocessor.clean)
    df["text_clean"] = df["text"].fillna("").apply(preprocessor.clean)
    return df


def train_pipeline(
    df: pd.DataFrame,
    pipeline_path: Path = PIPELINE_OUT,
    matrix_path: Path = MATRIX_OUT,
    use_spacy: bool = False,
) -> FeaturePipeline:
    """Clean, fit, and persist the unified feature pipeline."""
    df = clean_dataframe(df, use_spacy=use_spacy)

    titles = df["title_clean"].tolist()
    texts = df["text_clean"].tolist()
    labels = df["label"].tolist()

    pipeline = FeaturePipeline()
    pipeline.fit(titles, texts)

    pipeline.save(pipeline_path)

    matrix = pipeline.transform(titles, texts)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    try:

        # pyrefly: ignore[missing-import]
        from scipy.sparse import save_npz

        save_npz(str(matrix_path), matrix)
    except ImportError:
        
        # pyrefly: ignore[missing-import]
        import numpy as np

        np.save(str(matrix_path), matrix.toarray())

    print(
        f"Feature pipeline fitted on {len(titles)} documents "
        f"with {pipeline.n_features} features. Saved to {pipeline_path}"
    )
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist the unified feature pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--pipeline-out",
        type=Path,
        default=PIPELINE_OUT,
        help="Output path for the fitted pipeline joblib",
    )
    parser.add_argument(
        "--matrix-out",
        type=Path,
        default=MATRIX_OUT,
        help="Output path for the full feature matrix (npz)",
    )
    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy lemmatization during cleaning (requires en_core_web_sm)",
    )
    args = parser.parse_args()

    df = load_data(args.data_dir)
    train_pipeline(
        df,
        pipeline_path=args.pipeline_out,
        matrix_path=args.matrix_out,
        use_spacy=args.use_spacy,
    )


if __name__ == "__main__":
    main()