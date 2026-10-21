"""Unified feature engineering pipeline combining TF-IDF and handcrafted features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from backend.app.utils.text_features import (
    extract_all_stylometric_features,
    stylometric_feature_names,
)
from backend.app.utils.tfidf_vectorizer import (
    TfidfVectorizerWrapper,
    combine_title_text,
)

DEFAULT_VECTORIZER_PATH = Path(
    "backend/ml/models/vectorizers/tfidf_vectorizer.joblib"
)
DEFAULT_PIPELINE_PATH = Path(
    "backend/ml/models/saved/feature_pipeline.joblib"
)


class FeaturePipeline:
    """Fit TF-IDF on title+content and stack with stylometric features."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 3),
        sublinear_tf: bool = True,
    ) -> None:
        self.tfidf = TfidfVectorizerWrapper(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )
        self._fitted = False
        self._stylometric_names: List[str] = []

    def fit(self, titles: List[str], texts: List[str]) -> "FeaturePipeline":
        """Fit TF-IDF on combined title+text and record stylometric feature names."""
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")
        if not titles:
            raise ValueError("Cannot fit on empty input")

        docs = combine_title_text(titles, texts)
        self.tfidf.fit(docs)
        self._stylometric_names = stylometric_feature_names()
        self._fitted = True
        return self

    def transform(self, titles: List[str], texts: List[str]) -> csr_matrix:
        """Transform cleaned titles/texts into a combined sparse feature matrix."""
        if not self._fitted:
            raise RuntimeError("FeaturePipeline has not been fitted yet. Call fit() first.")
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")

        docs = combine_title_text(titles, texts)
        tfidf_matrix = self.tfidf.transform(docs)

        stylometric_rows = []
        for title, body in zip(titles, texts):
            feats = extract_all_stylometric_features(f"{title} {body}")
            stylometric_rows.append([feats[name] for name in self._stylometric_names])

        stylometric_matrix = csr_matrix(
            np.array(stylometric_rows, dtype=np.float64)
        )

        return hstack([tfidf_matrix, stylometric_matrix], format="csr")

    def fit_transform(self, titles: List[str], texts: List[str]) -> csr_matrix:
        """Fit on titles/texts and return the combined feature matrix."""
        self.fit(titles, texts)
        return self.transform(titles, texts)

    def save(self, path: Path | str = DEFAULT_PIPELINE_PATH) -> Path:
        """Persist the fitted pipeline to disk using joblib."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PIPELINE_PATH) -> "FeaturePipeline":
        """Load a previously persisted pipeline from disk."""
        return joblib.load(Path(path))

    @property
    def n_features(self) -> int:
        """Total number of features (TF-IDF + stylometric)."""
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fitted yet")
        return int(self.tfidf.n_features + len(self._stylometric_names))

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of all feature names."""
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fitted yet")
        return list(self.tfidf.feature_names_) + list(self._stylometric_names)