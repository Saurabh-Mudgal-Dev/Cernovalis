"""Tests for the POST /api/v1/analyze endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import backend.app.main as main_module
from backend.app.main import app
from backend.app.services.predictor import (
    EnsembleArtifacts,
    PredictionResult,
    PredictorService,
    load_predictor,
)

SAVED_DIR = Path("backend/ml/models/saved")
REAL_ARTIFACTS_PRESENT = (
    (SAVED_DIR / "logistic_regression.joblib").exists()
    and Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib").exists()
)


class StubPredictor:
    """Predictor returning a canned result, with a real (tiny) explainer backend."""

    def __init__(self, result, ready=True):
        self.result = result
        self.ready = ready
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["good news today", "bad shocking scandal"])
        self.artifacts = EnsembleArtifacts(
            vectorizer=vectorizer,
            classical={
                "logistic_regression": LogisticRegression().fit(matrix, [1, 0])
            },
            member_order=["logistic_regression"],
        )

    def predict(self, title=None, text=None):
        if (title or "").strip() == "":
            if text is None or not text.strip():
                raise ValueError("At least one of 'title' or 'text' is required.")
        if self.result is None:
            raise ValueError("stub failure")
        return self.result


def make_result(confidence=0.9, probability_real=0.9, verdict="REAL"):
    return PredictionResult(
        verdict=verdict,
        confidence=confidence,
        probability_real=probability_real,
        model_probabilities={"logistic_regression": probability_real},
    )


def install_stub(monkeypatch, predictor):
    monkeypatch.setattr(
        main_module, "_load_predictor", lambda settings: (predictor, [])
    )


def test_analyze_text_returns_200_and_serializes_explanation(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze", json={"text": "Retractions are " "issued today."}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "REAL"
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["risk_level"] == "LOW"
    assert payload["input_type"] == "text"
    assert payload["original_input"]
    assert "logistic_regression" in payload["explanation"]["model_contributions"]
    assert payload["source_credibility"]["label"] == "UNKNOWN"


def test_analyze_fake_verdict_maps_to_risk_level(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result(verdict="FAKE", probability_real=0.1)))

    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json={"text": "Sensational garbage."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "FAKE"
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["risk_level"] == "HIGH"


def test_analyze_rejects_url_until_scraper_lands(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze", json={"url": "https://example.com/article"}
        )

    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


def test_analyze_returns_503_when_models_are_missing(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result(), ready=False))

    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json={"text": "Any article."})

    assert response.status_code == 503


def test_analyze_converts_processing_failures_to_422(monkeypatch):
    install_stub(monkeypatch, StubPredictor(None))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze", json={"text": "the and of were at."}
        )

    assert response.status_code == 422


@pytest.mark.skipif(not REAL_ARTIFACTS_PRESENT, reason="trained model bundle not present")
def test_analyze_with_real_artifacts_end_to_end():
    # Real artifacts, loaded through the normal lifespan path (no stubbing).
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": (
                    "WASHINGTON (Reuters) - The Federal Reserve left interest rates "
                    "unchanged on Wednesday and reiterated its data-dependent stance, "
                    "citing moderating inflation and steady job growth."
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] in ("REAL", "FAKE")
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["explanation"]["model_contributions"]
    assert payload["explanation"]["top_features"]
