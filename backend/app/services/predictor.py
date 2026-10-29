"""Model loading, preprocessing and ensemble inference for the prediction API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
from loguru import logger

from backend.app.services.preprocessor import TextPreprocessor
from backend.app.utils.ensemble import (
    get_probabilities,
    soft_vote,
    stacked_matrix,
)

REAL_THRESHOLD = 0.5
VERDICT_REAL = "REAL"
VERDICT_FAKE = "FAKE"

DEFAULT_MAX_LEN = 300
BILSTM_MEMBER = "bilstm"

DEFAULT_MODEL_DIR = Path("backend/ml/models/saved")
DEFAULT_VECTORIZER_PATH = Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib")

CLASSICAL_MEMBERS = ("logistic_regression", "naive_bayes", "random_forest", "xgboost")
MODEL_FILES: Mapping[str, str] = {
    "logistic_regression": "logistic_regression.joblib",
    "naive_bayes": "naive_bayes.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost_model.joblib",
}
BILSTM_MODEL_FILE = "bilstm_model.keras"
BILSTM_TOKENIZER_FILE = "bilstm_tokenizer.json"
META_CLASSIFIER_FILE = "meta_classifier.joblib"
ENSEMBLE_REPORT_FILE = "ensemble_report.json"
BILSTM_REPORT_FILE = "bilstm_report.json"


class PredictorNotReadyError(RuntimeError):
    """Raised when inference is attempted without the required artifacts."""


@dataclass
class ArtifactStatus:
    """Load state of a single artifact for health reporting."""

    name: str
    loaded: bool
    detail: str | None = None


@dataclass
class EnsembleArtifacts:
    """The loaded model artifacts; individual members may be absent."""

    vectorizer: Any | None = None
    classical: dict[str, Any] = field(default_factory=dict)
    bilstm: Any | None = None
    word_index: dict[str, int] = field(default_factory=dict)
    max_len: int = DEFAULT_MAX_LEN
    meta_classifier: Any | None = None
    member_order: list[str] = field(default_factory=list)
    statuses: list[ArtifactStatus] = field(default_factory=list)


@dataclass
class PredictionResult:
    """Outcome of one inference pass over an article."""

    verdict: str
    confidence: float
    probability_real: float
    model_probabilities: dict[str, float]


class PredictorService:
    """Runs the preprocessing → features → ensemble → meta-classifier pipeline."""

    def __init__(self, artifacts: EnsembleArtifacts) -> None:
        self.artifacts = artifacts
        # Same settings as the shared training pipeline (train_baselines.load_dataset).
        self._preprocessor = TextPreprocessor(use_spacy=False, remove_stopwords=True)

    @property
    def ready(self) -> bool:
        """True when the feature extractor is available."""
        return self.artifacts.vectorizer is not None

    @property
    def statuses(self) -> list[ArtifactStatus]:
        return list(self.artifacts.statuses)

    def predict(self, title: str | None = None, text: str | None = None) -> PredictionResult:
        """Classify a raw title/text pair and return the ensemble decision."""
        if not self.ready:
            raise PredictorNotReadyError(
                "The TF-IDF vectorizer is not loaded; check the models directory."
            )

        title_clean = (title or "").strip()
        text_clean = (text or "").strip()
        if not title_clean and not text_clean:
            raise ValueError("At least one of 'title' or 'text' is required.")

        document = f"{self._preprocessor.clean(title_clean)} {self._preprocessor.clean(text_clean)}".strip()
        if not document:
            raise ValueError("Preprocessing produced an empty document.")

        model_probabilities: dict[str, float] = {}
        item_matrix = self.artifacts.vectorizer.transform([document])
        for name, model in self.artifacts.classical.items():
            model_probabilities[name] = float(get_probabilities(model, item_matrix)[0])

        if self.artifacts.bilstm is not None and self.artifacts.word_index:
            from backend.app.utils.bilstm import predict_probs, sequences_to_ids, tokenize

            sequences = sequences_to_ids(
                tokenize([document]),
                self.artifacts.word_index,
                max_len=self.artifacts.max_len,
            )
            model_probabilities[BILSTM_MEMBER] = float(
                predict_probs(self.artifacts.bilstm, sequences)[0]
            )

        if not model_probabilities:
            raise PredictorNotReadyError("No ensemble member models were loaded.")

        probability_real = self._ensemble_probability(model_probabilities)
        return PredictionResult(
            verdict=VERDICT_REAL if probability_real >= REAL_THRESHOLD else VERDICT_FAKE,
            confidence=max(probability_real, 1.0 - probability_real),
            probability_real=probability_real,
            model_probabilities=model_probabilities,
        )

    def _ensemble_probability(
        self, model_probabilities: Mapping[str, float]
    ) -> float:
        """Combine member probabilities via the meta-classifier, else soft voting."""
        ordered_names = [
            name for name in self.artifacts.member_order if name in model_probabilities
        ]
        meta = self.artifacts.meta_classifier
        if meta is not None and ordered_names == self.artifacts.member_order:
            stacked = stacked_matrix(
                {name: [model_probabilities[name]] for name in ordered_names}
            )
            return float(get_probabilities(meta, stacked)[0])

        available = {
            name: np.asarray([model_probabilities[name]]) for name in ordered_names
        }
        return float(soft_vote(available)[0])


def _load_tfidf_vectorizer(path: Path) -> Any:
    from backend.app.utils.tfidf_vectorizer import TfidfVectorizerWrapper

    return TfidfVectorizerWrapper.load(path)


def _load_optional_artifact(
    statuses: list[ArtifactStatus],
    name: str,
    path: Path | None,
    loader: Callable[[], Any],
    assign: Callable[[Any], None],
) -> None:
    """Attempt one artifact load and record its status regardless of outcome."""
    if path is not None and not path.exists():
        statuses.append(ArtifactStatus(name=name, loaded=False, detail=str(path)))
        logger.warning("{} artifact not found at {}", name, path)
        return
    try:
        artifact = loader()
        assign(artifact)
        statuses.append(ArtifactStatus(name=name, loaded=True, detail=str(path)))
        logger.info("Loaded {} from {}", name, path)
    except Exception as error:  # noqa: BLE001 - one bad artifact must not block startup
        statuses.append(
            ArtifactStatus(
                name=name,
                loaded=False,
                detail=f"{type(error).__name__}: {error}",
            )
        )
        logger.exception("Failed to load {} artifact {}", name, path)


def _member_order(model_dir: Path, loaded_members: list[str]) -> list[str]:
    """The column order the meta-classifier was trained with, else training defaults."""
    report_path = model_dir / ENSEMBLE_REPORT_FILE
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                coefficients = json.load(handle).get("meta_model_coefficients", {})
            if coefficients:
                return list(coefficients)
        except (OSError, ValueError):
            logger.warning("Could not read member order from {}", report_path)
    fallback = list(CLASSICAL_MEMBERS) + [BILSTM_MEMBER]
    return [name for name in fallback if name in loaded_members]


def load_predictor(settings) -> tuple[PredictorService, list[ArtifactStatus]]:
    """Load every available artifact into a PredictorService; absent files are tolerated.

    `settings` must expose `model_dir` and `vectorizer_path` path attributes.
    Each artifact loads independently so one damaged file cannot prevent startup.
    """
    model_dir = Path(settings.model_dir)
    service = PredictorService(artifacts=EnsembleArtifacts())

    vectorizer_path = Path(settings.vectorizer_path)
    _load_optional_artifact(
        statuses=service.artifacts.statuses,
        name="tfidf_vectorizer",
        path=vectorizer_path,
        loader=lambda: _load_tfidf_vectorizer(vectorizer_path),
        assign=lambda artifact: setattr(service.artifacts, "vectorizer", artifact),
    )

    for name, filename in MODEL_FILES.items():
        path = model_dir / filename
        _load_optional_artifact(
            statuses=service.artifacts.statuses,
            name=name,
            path=path,
            loader=lambda p=path: joblib.load(p),
            assign=lambda artifact, member=name: service.artifacts.classical.update(
                {member: artifact}
            ),
        )

    bilstm_path = model_dir / BILSTM_MODEL_FILE
    if bilstm_path.exists():
        _load_optional_artifact(
            statuses=service.artifacts.statuses,
            name=BILSTM_MEMBER,
            path=bilstm_path,
            loader=lambda: _load_bilstm(bilstm_path),
            assign=lambda artifact: setattr(service.artifacts, "bilstm", artifact),
        )
        tokenizer_path = model_dir / BILSTM_TOKENIZER_FILE
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, "r", encoding="utf-8") as handle:
                    service.artifacts.word_index = {
                        word: int(index) for word, index in json.load(handle).items()
                    }
            except (OSError, ValueError) as error:
                logger.warning("Failed to read BiLSTM tokenizer {}: {}", tokenizer_path, error)
        service.artifacts.max_len = _read_max_len(model_dir)
    else:
        service.artifacts.statuses.append(
            ArtifactStatus(name=BILSTM_MEMBER, loaded=False, detail=str(bilstm_path))
        )

    _load_optional_artifact(
        statuses=service.artifacts.statuses,
        name="meta_classifier",
        path=model_dir / META_CLASSIFIER_FILE,
        loader=lambda: joblib.load(model_dir / META_CLASSIFIER_FILE),
        assign=lambda artifact: setattr(service.artifacts, "meta_classifier", artifact),
    )

    loaded_members = list(service.artifacts.classical)
    if service.artifacts.bilstm is not None and service.artifacts.word_index:
        loaded_members.append(BILSTM_MEMBER)
    service.artifacts.member_order = _member_order(model_dir, loaded_members)

    loaded_count = sum(1 for status in service.artifacts.statuses if status.loaded)
    logger.info(
        "Predictor ready: {}/{} artifacts loaded", loaded_count, len(service.artifacts.statuses)
    )
    return service, service.artifacts.statuses


def _load_bilstm(path: Path) -> Any:
    from backend.app.utils.bilstm import load_bilstm

    return load_bilstm(str(path))


def _read_max_len(model_dir: Path) -> int:
    """Prefer the max_len recorded in the BiLSTM training report."""
    report_path = model_dir / BILSTM_REPORT_FILE
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                tokenization = json.load(handle).get("tokenization", {})
            return int(tokenization.get("max_len", DEFAULT_MAX_LEN))
        except (OSError, ValueError):
            logger.warning("Could not read max_len from {}", report_path)
    return DEFAULT_MAX_LEN
