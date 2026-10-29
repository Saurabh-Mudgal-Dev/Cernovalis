"""Tests for the predictor service: loading, preprocessing and inference."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.app.services.predictor import (
    ArtifactStatus,
    EnsembleArtifacts,
    PredictorNotReadyError,
    PredictorService,
    load_predictor,
)


@dataclass
class StubSettings:
    model_dir: Path = Path("backend/ml/models/saved")
    vectorizer_path: Path = Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib")


class FakeVectorizer:
    def __init__(self):
        self.docs = None

    def transform(self, docs):
        self.docs = list(docs)
        return np.zeros((len(docs), 4))


class FakeSklearnModel:
    """Sklearn-style model reporting a fixed P(real)."""

    def __init__(self, p_real):
        self.p_real = p_real

    def predict_proba(self, X):
        p = np.full((len(X),), self.p_real)
        return np.column_stack([1 - p, p])


class FakeKerasModel:
    """Keras-style model (predict output shaped (n, 1)) reporting P(real)."""

    def __init__(self, p_real):
        self.p_real = p_real
        self.sequences = None

    def predict(self, X, verbose=0):
        self.sequences = np.asarray(X)
        return np.full((len(self.sequences), 1), self.p_real)


class FakeMeta:
    """Meta-classifier averaging the stacked member probabilities."""

    def __init__(self):
        self.last_stacked = None

    def predict_proba(self, stacked):
        self.last_stacked = np.asarray(stacked)
        mean = np.mean(self.last_stacked, axis=1)
        return np.column_stack([1 - mean, mean])


def make_service(artifacts, member_order=None):
    if member_order is None:
        member_order = list(artifacts.classical) + (
            ["bilstm"] if artifacts.bilstm is not None else []
        )
    artifacts.member_order = member_order
    return PredictorService(artifacts=artifacts)


def test_predict_requires_vectorizer_and_non_empty_input():
    artifacts = EnsembleArtifacts(
        vectorizer=None,
        classical={"logistic_regression": FakeSklearnModel(0.5)},
    )
    service = make_service(artifacts)

    with pytest.raises(PredictorNotReadyError):
        service.predict(text="some article")

    ready_artifacts = EnsembleArtifacts(
        vectorizer=FakeVectorizer(),
        classical={"logistic_regression": FakeSklearnModel(0.5)},
    )
    ready_service = make_service(ready_artifacts)
    with pytest.raises(ValueError):
        ready_service.predict(title="   ", text=None)


def test_predict_combines_preprocessed_title_and_text():
    vectorizer = FakeVectorizer()
    service = make_service(
        EnsembleArtifacts(
            vectorizer=vectorizer,
            classical={"logistic_regression": FakeSklearnModel(0.9)},
        ),
        member_order=["logistic_regression"],
    )

    result = service.predict(
        title="Mysterious Object!!!", text="Scientists discovered a spaceship."
    )

    assert result.verdict == "REAL"
    assert result.confidence == pytest.approx(0.9)
    assert result.probability_real == pytest.approx(0.9)
    assert vectorizer.docs == ["mysterious object scientists discovered spaceship"]


def test_predict_uses_meta_classifier_in_report_member_order():
    meta = FakeMeta()
    service = make_service(
        EnsembleArtifacts(
            vectorizer=FakeVectorizer(),
            classical={
                "logistic_regression": FakeSklearnModel(0.2),
                "naive_bayes": FakeSklearnModel(0.8),
            },
            meta_classifier=meta,
        ),
        member_order=["naive_bayes", "logistic_regression"],
    )

    result = service.predict(text="A neutral article about weather.")

    # Columns must be stacked in the meta-classifier's trained column order.
    assert meta.last_stacked.tolist() == [[0.8, 0.2]]
    assert result.probability_real == pytest.approx(0.5)
    assert result.verdict == "REAL"
    assert result.model_probabilities["naive_bayes"] == pytest.approx(0.8)


def test_predict_falls_back_to_soft_vote_without_meta():
    service = make_service(
        EnsembleArtifacts(
            vectorizer=FakeVectorizer(),
            classical={
                "logistic_regression": FakeSklearnModel(0.2),
                "random_forest": FakeSklearnModel(0.6),
            },
        ),
        member_order=["logistic_regression", "random_forest"],
    )

    result = service.predict(text="A neutral article about weather.")

    assert result.probability_real == pytest.approx(0.4)
    assert result.verdict == "FAKE"


def test_bilstm_member_is_tokenized_through_word_index():
    keras = FakeKerasModel(0.9)
    service = make_service(
        EnsembleArtifacts(
            vectorizer=FakeVectorizer(),
            classical={"logistic_regression": FakeSklearnModel(0.1)},
            bilstm=keras,
            word_index={"alpha": 1, "beta": 2},
            max_len=6,
        ),
        member_order=["logistic_regression", "bilstm"],
    )

    result = service.predict(text="alpha beta gamma")

    assert keras.sequences.shape == (1, 6)
    assert keras.sequences[0].tolist() == [1, 2, 0, 0, 0, 0]
    assert result.model_probabilities["bilstm"] == pytest.approx(0.9)


def test_bilstm_is_skipped_without_word_index():
    service = make_service(
        EnsembleArtifacts(
            vectorizer=FakeVectorizer(),
            classical={"logistic_regression": FakeSklearnModel(0.6)},
            bilstm=FakeKerasModel(0.9),
            word_index={},
        ),
        member_order=["logistic_regression", "bilstm"],
    )

    result = service.predict(text="A neutral article about weather.")

    assert "bilstm" not in result.model_probabilities
    assert result.probability_real == pytest.approx(0.6)


def test_load_predictor_tolerates_missing_artifacts(tmp_path):
    settings = StubSettings(
        model_dir=tmp_path,
        vectorizer_path=tmp_path / "does_not_exist.joblib",
    )
    service, statuses = load_predictor(settings)

    assert service.ready is False
    names = {status.name: status for status in statuses}
    assert names["tfidf_vectorizer"].loaded is False
    assert names["logistic_regression"].loaded is False
    assert names["bilstm"].loaded is False
    assert names["meta_classifier"].loaded is False
    with pytest.raises(PredictorNotReadyError):
        service.predict(text="anything")


def test_load_predictor_partially_loads_bundle(tmp_path):
    # Minimal but real sklearn artifacts exercising the actual loading path.
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    sklearn_tfidf = TfidfVectorizer().fit(["good news today", "bad news tomorrow"])
    joblib.dump(sklearn_tfidf, tmp_path / "tfidf_vectorizer.joblib")

    # The classical model must accept the vectorizer's output width.
    matrix = sklearn_tfidf.transform(["good news today", "bad news tomorrow"])
    joblib.dump(LogisticRegression().fit(matrix, [1, 0]), tmp_path / "logistic_regression.joblib")

    # Only logistic_regression is present, so the meta stack is one column wide.
    meta = LogisticRegression().fit([[0.2], [0.9]], [0, 1])
    joblib.dump(meta, tmp_path / "meta_classifier.joblib")

    @dataclass
    class BundleSettings:
        model_dir: Path = tmp_path
        vectorizer_path: Path = tmp_path / "tfidf_vectorizer.joblib"

    service, statuses = load_predictor(BundleSettings())

    loaded = {status.name for status in statuses if status.loaded}
    assert loaded == {"tfidf_vectorizer", "logistic_regression", "meta_classifier"}
    assert service.ready is True
    assert service.artifacts.member_order == ["logistic_regression"]
    assert "bilstm" not in service.artifacts.member_order

    result = service.predict(title="good", text="news today")
    assert set(result.model_probabilities) == {"logistic_regression"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.verdict in ("REAL", "FAKE")
    assert result.probability_real == pytest.approx(result.confidence)


SAVED_DIR = Path("backend/ml/models/saved")
ARTIFACTS_PRESENT = (
    (SAVED_DIR / "logistic_regression.joblib").exists()
    and (SAVED_DIR / "meta_classifier.joblib").exists()
    and Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib").exists()
)


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="trained model bundle not present")
def test_load_predictor_real_artifacts_end_to_end():
    service, statuses = load_predictor(StubSettings())

    loaded_names = {status.name for status in statuses if status.loaded}
    assert loaded_names >= {
        "tfidf_vectorizer",
        "logistic_regression",
        "naive_bayes",
        "random_forest",
        "xgboost",
        "bilstm",
        "meta_classifier",
    }

    result = service.predict(
        title="Final results",
        text=(
            "WASHINGTON (Reuters) - The Federal Reserve left interest rates "
            "unchanged on Wednesday and reiterated its data-dependent stance, "
            "citing moderating inflation and steady job growth."
        ),
    )
    assert result.verdict == "REAL"
    assert result.confidence >= 0.7
    assert result.probability_real == pytest.approx(result.confidence)
    assert 0.0 <= result.probability_real <= 1.0

    fake_result = service.predict(
        title="BREAKING: SHOCKING secret revealed",
        text=(
            "They do not want you to know this!! Click here now before it is "
            "deleted forever. Doctors HATE this one weird trick!!"
        ),
    )
    assert fake_result.verdict == "FAKE"
    assert fake_result.confidence >= 0.7
    assert set(fake_result.model_probabilities) <= {
        "logistic_regression",
        "naive_bayes",
        "random_forest",
        "xgboost",
        "bilstm",
    }
