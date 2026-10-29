"""Tests for the POST /api/v1/batch CSV endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import backend.app.main as main_module
from backend.app.main import app
from backend.app.services.predictor import (
    EnsembleArtifacts,
    PredictionResult
)

SAVED_DIR = Path("backend/ml/models/saved")
REAL_ARTIFACTS_PRESENT = (
    (SAVED_DIR / "logistic_regression.joblib").exists()
    and Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib").exists()
)


class StubPredictor:
    """Predictor with a canned result plus a real (tiny) explainer backend."""

    def __init__(self, result, ready=True):
        self.result = result
        self.ready = ready
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["good news today", "bad shocking scandal"])
        self.artifacts = EnsembleArtifacts(
            vectorizer=vectorizer,
            classical={"logistic_regression": LogisticRegression().fit(matrix, [1, 0])},
            member_order=["logistic_regression"],
        )

    def predict(self, title=None, text=None):
        text = text or ""
        if not text.strip() and not (title or "").strip():
            raise ValueError("Row has no title and no article text.")
        return self.result


def make_result(probability_real=0.9):
    return PredictionResult(
        verdict="REAL",
        confidence=0.9,
        probability_real=probability_real,
        model_probabilities={"logistic_regression": probability_real},
    )


def batch_payload():
    # Third row is completely empty except for the separators: it must fail.
    return "title,text\nReport,Steady rate decision at the meeting.\n,\nSensational,!!!"


def install_stub(monkeypatch, predictor):
    monkeypatch.setattr(main_module, "_load_predictor", lambda settings: (predictor, []))


def test_batch_processes_rows_and_reports_failures(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", batch_payload().encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["succeeded"] == 2
    assert payload["failed"] == 1
    assert payload["results"][0]["row"] == 1
    assert payload["results"][1]["row"] == 2
    assert payload["results"][1]["result"] is None
    assert "no title" in payload["results"][1]["error"]
    succeeded_rows = [
        item for item in payload["results"] if item["result"] is not None
    ]
    assert all(item["result"]["verdict"] == "REAL" for item in succeeded_rows)
    assert all(
        item["result"]["explanation"]["model_contributions"]
        for item in succeeded_rows
    )


def test_batch_requires_text_column(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", b"title,summary\nA,B\n", "text/csv")},
        )

    assert response.status_code == 422
    assert "'text'" in response.json()["detail"]


def test_batch_rejects_empty_file_and_garbage(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))

    with TestClient(app) as client:
        empty = client.post(
            "/api/v1/batch", files={"file": ("rows.csv", b"", "text/csv")}
        )
        header_only = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", b"title,text\n", "text/csv")},
        )

    assert empty.status_code == 422
    assert header_only.status_code == 200
    assert header_only.json()["total"] == 0


def test_batch_rejects_oversized_uploads(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result()))
    csv_text = "title,text\n" + "Report,Body text.\n" * 1001

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", csv_text.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 422
    assert "1000-row limit" in response.json()["detail"]


def test_batch_returns_503_when_models_are_missing(monkeypatch):
    install_stub(monkeypatch, StubPredictor(make_result(), ready=False))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", b"title,text\nA,B\n", "text/csv")},
        )

    assert response.status_code == 503


@pytest.mark.skipif(not REAL_ARTIFACTS_PRESENT, reason="trained model bundle not present")
def test_batch_with_real_artifacts_end_to_end():
    csv_text = (
        "title,text\n"
        "Final results,The Federal Reserve left rates unchanged on Wednesday"
        " citing moderating inflation and steady job growth, as reported by Reuters.\n"
        "BREAKING,SHOCKING secret revealed!! Doctors hate this one weird trick!!\n"
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/batch",
            files={"file": ("rows.csv", csv_text.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["succeeded"] == 2
    for item in payload["results"]:
        assert item["result"]["verdict"] in ("REAL", "FAKE")
        assert item["result"]["explanation"]["top_features"]
