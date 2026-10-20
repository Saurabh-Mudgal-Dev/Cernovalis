"""TF-IDF vectorizer wrapper for Cernovalis fake-news detection.

Trains a scikit-learn ``TfidfVectorizer`` on cleaned article titles and body
text, then persists it for reuse by both training notebooks and the FastAPI
inference API.

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

# pyrefly: ignore[missing-import]
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# PLAN.md Section 4.B TF-IDF configuration
DEFAULT_MAX_FEATURES = 10000
DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_SUBLINEAR_TF = True
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 1.0

# Default persistence paths (PLAN.md backend config)
DEFAULT_VECTORIZER_PATH = Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib")


class TfidfVectorizerWrapper:
    """Thin wrapper around scikit-learn's TfidfVectorizer with persistence."""

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
        sublinear_tf: bool = DEFAULT_SUBLINEAR_TF,
        min_df: Union[int, float] = DEFAULT_MIN_DF,
        max_df: Union[int, float] = DEFAULT_MAX_DF,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            lowercase=False,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "TfidfVectorizerWrapper":
        """Fit the vectorizer on an iterable of cleaned text documents."""
        if not texts:
            raise ValueError("Cannot fit TF-IDF vectorizer on empty text list")
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]):
        """Transform cleaned text documents into a TF-IDF sparse matrix."""
        if not self._fitted:
            raise RuntimeError(
                "TF-IDF vectorizer has not been fitted yet. Call fit() first."
            )
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit on ``texts`` and return the resulting TF-IDF sparse matrix."""
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix

    def save(self, path: Path | str = DEFAULT_VECTORIZER_PATH) -> Path:
        """Persist the fitted vectorizer to disk using joblib."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted vectorizer")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_VECTORIZER_PATH) -> "TfidfVectorizerWrapper":
        """Load a previously persisted vectorizer from disk.
        """
        wrapper = cls()
        loaded = joblib.load(Path(path))
        wrapper.vectorizer = loaded
        wrapper._fitted = True
        # Mirror the sklearn vectorizer configuration on the wrapper.
        wrapper.max_features = getattr(loaded, "max_features", wrapper.max_features)
        wrapper.ngram_range = getattr(loaded, "ngram_range", wrapper.ngram_range)
        wrapper.sublinear_tf = getattr(loaded, "sublinear_tf", wrapper.sublinear_tf)
        return wrapper

    @property
    def feature_names_(self) -> List[str]:
        """Return the feature names learned during fitting."""
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted yet")
        return list(self.vectorizer.get_feature_names_out())

    @property
    def n_features(self) -> int:
        """Return the number of features in the fitted vectorizer."""
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted yet")
        return int(len(self.feature_names_))


def combine_title_text(titles: List[str], texts: List[str]) -> List[str]:
    """Combine cleaned titles and article bodies into a single document per row.

    Title is weighted implicitly by appearing first; downstream callers may
    alternatively repeat the title to boost its influence.
    """
    if len(titles) != len(texts):
        raise ValueError("titles and texts must have the same length")
    return [
        f"{title} {body}".strip() for title, body in zip(titles, texts)
    ]


def split_train_test(
    titles: List[str],
    texts: List[str],
    labels: List[int],
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:
    """Stratified train/test split following (70/15/15 overall; here 70/30
    for the TF-IDF fitting split, with the remaining 30% reserved for evaluation
    by the caller)."""
    (
        train_titles,
        test_titles,
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = train_test_split(
        titles,
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return (
        train_titles,
        test_titles,
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    )