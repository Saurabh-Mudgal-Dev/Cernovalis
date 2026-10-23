"""Feature scaler wrapper for Cernovalis fake-news detection."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

DEFAULT_SCALER_PATH = Path("backend/ml/models/vectorizers/feature_scaler.joblib")


class FeatureScaler:
    """Standard scaler for dense stylometric feature vectors."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "FeatureScaler":
        """Fit the scaler on a dense feature matrix."""
        if X.size == 0:
            raise ValueError("Cannot fit scaler on empty matrix")
        self.scaler.fit(X)
        self._feature_names = list(feature_names) if feature_names else None
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform a dense feature matrix using fitted scaler parameters."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on X and return the scaled matrix."""
        self.fit(X, feature_names)
        return self.transform(X)

    def save(self, path: Path | str = DEFAULT_SCALER_PATH) -> Path:
        """Persist the fitted scaler to disk using joblib."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted scaler")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SCALER_PATH) -> "FeatureScaler":
        """Load a previously persisted scaler from disk."""
        scaler = cls()
        scaler.scaler = joblib.load(Path(path))
        scaler._fitted = True
        return scaler

    @property
    def n_features(self) -> int:
        """Number of features the scaler was fitted on."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        return int(self.scaler.n_features_in_)

    @property
    def means(self) -> np.ndarray:
        """Per-feature mean learned during fitting."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        return self.scaler.mean_

    @property
    def scales(self) -> np.ndarray:
        """Per-feature scale learned during fitting."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        return self.scaler.scale_