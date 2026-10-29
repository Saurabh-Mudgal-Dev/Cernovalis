"""Tests for the explainability service."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.app.services.explainer import (
    build_explanation,
    linguistic_flags,
    model_contributions,
    risk_level_for,
    stylometric_highlights,
    top_token_features,
)
from backend.app.services.predictor import EnsembleArtifacts, PredictionResult


def make_tfidf_lr():
    docs = ["good news today", "bad shocking scandal"]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)
    model = LogisticRegression().fit(matrix, [1, 0])
    return vectorizer, model


def make_prediction(probabilities):
    p = float(np.mean(list(probabilities.values())))
    return PredictionResult(
        verdict="REAL" if p >= 0.5 else "FAKE",
        confidence=max(p, 1 - p),
        probability_real=p,
        model_probabilities=probabilities,
    )


def test_model_contributions_map_probabilities_to_verdicts():
    prediction = make_prediction(
        {"logistic_regression": 0.9, "random_forest": 0.2}
    )

    contributions = model_contributions(prediction)

    assert contributions["logistic_regression"] == {
        "prediction": "REAL",
        "confidence": pytest.approx(0.9),
    }
    assert contributions["random_forest"] == {
        "prediction": "FAKE",
        "confidence": pytest.approx(0.8),
    }


@pytest.mark.parametrize(
    ("verdict", "confidence", "expected"),
    [
        ("REAL", 0.99, "LOW"),
        ("FAKE", 0.95, "HIGH"),
        ("FAKE", 0.75, "MEDIUM"),
        ("FAKE", 0.51, "LOW"),
    ],
)
def test_risk_level_for(verdict, confidence, expected):
    assert risk_level_for(verdict, confidence) == expected


def test_linguistic_flags_detect_stylistic_red_flags():
    clicking = "CALL ALL TRUE BRAND NEW SHOCKING now " * 5
    flags = linguistic_flags(clicking + "!!! " * 20)

    assert any("capitalization" in flag.lower() for flag in flags)
    assert any("shouting" in flag for flag in flags)


def test_linguistic_flags_stay_silent_on_plain_text():
    plain = 'The committee said "the results are final" after reviewing the data. ' * 2
    assert linguistic_flags(plain) == []


def test_top_token_features_rank_tokens_by_lr_impact():
    vectorizer, model = make_tfidf_lr()

    top = top_token_features("good news today", model, vectorizer)

    names = [entry["feature"] for entry in top]
    assert names
    assert all(name.startswith("tfidf:") for name in names)
    assert "tfidf:good" in names
    assert top[0]["impact"] == "high"
    impacts = {entry["impact"] for entry in top}
    assert impacts <= {"high", "medium", "low"}


def test_top_token_features_return_empty_without_model():
    vectorizer, _ = make_tfidf_lr()

    assert top_token_features("anything", None, None) == []
    assert top_token_features("anything", object(), vectorizer) == []


def test_stylometric_highlights_cover_expected_features():
    features = {
        "uppercase_char_ratio": 0.5,
        "exclamation_density": 0.0,
        "question_density": 0.02,
        "quotation_density": 0.0,
        "subjectivity": 0.4,
        "flesch_kincaid_grade": 3.0,
        "type_token_ratio": 0.9,
    }

    highlights = stylometric_highlights(features)

    assert {entry["feature"] for entry in highlights} == set(features)
    caps = next(entry for entry in highlights if entry["feature"] == "uppercase_char_ratio")
    assert caps["impact"] == "high"
    assert caps["value"] == 0.5
    reading_ease = next(entry for entry in highlights if entry["feature"] == "flesch_kincaid_grade")
    assert reading_ease["impact"] == "low"


def test_build_explanation_assembles_all_sections():
    vectorizer, model = make_tfidf_lr()
    artifacts = EnsembleArtifacts(
        vectorizer=vectorizer,
        classical={"logistic_regression": model},
        member_order=["logistic_regression"],
    )
    prediction = make_prediction({"logistic_regression": 0.93})

    explanation = build_explanation(
        "Bad shocking scandal about good news today.", prediction, artifacts
    )

    assert explanation["model_contributions"]["logistic_regression"]["prediction"] == "REAL"
    assert explanation["model_contributions"]["logistic_regression"]["confidence"] == pytest.approx(0.93)
    assert any(entry["feature"].startswith("tfidf:") for entry in explanation["top_features"])
    assert any(
        entry["feature"] == "uppercase_char_ratio" for entry in explanation["top_features"]
    )
    assert isinstance(explanation["linguistic_flags"], list)
