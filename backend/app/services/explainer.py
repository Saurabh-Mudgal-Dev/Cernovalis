"""Explainability service: token contributions, stylometric flags, model breakdown."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from backend.app.services.preprocessor import TextPreprocessor
from backend.app.services.predictor import (
    EnsembleArtifacts,
    PredictionResult,
    REAL_THRESHOLD,
    VERDICT_REAL,
)
from backend.app.utils.text_features import extract_all_stylometric_features

DEFAULT_TOP_K_TOKENS = 8

# Flag thresholds over the stylometric feature values.
HIGH_CAPS_RATIO = 0.2
MIN_ALL_CAPS_WORDS = 3
HIGH_EXCLAMATION_DENSITY = 0.005
MIN_WORDS_FOR_QUOTATION_FLAG = 80
HIGH_SUBJECTIVITY = 0.3

FAKE_RISK_STEPS = ((0.9, "HIGH"), (0.65, "MEDIUM"))

_stylometric_highlights = (
    ("uppercase_char_ratio", 0.2, 0.1),
    ("exclamation_density", 0.005, 0.0025),
    ("question_density", 0.01, 0.005),
    ("quotation_density", 0.01, 0.005),
    ("subjectivity", 0.3, 0.15),
    ("flesch_kincaid_grade", 14.0, 9.0),
    ("type_token_ratio", None, None),
)

_shared_preprocessor = TextPreprocessor(use_spacy=False, remove_stopwords=True)


def risk_level_for(verdict: str, confidence: float) -> str:
    """Map verdict and confidence onto LOW/MEDIUM/HIGH risk."""
    if verdict != "FAKE":
        return "LOW"
    for threshold, level in FAKE_RISK_STEPS:
        if confidence >= threshold:
            return level
    return "LOW"


def model_contributions(
    prediction: PredictionResult,
) -> dict[str, dict[str, Any]]:
    """Per-model verdict and confidence relative to the winning class."""
    contributions: dict[str, dict[str, Any]] = {}
    for name, probability_real in prediction.model_probabilities.items():
        p = float(probability_real)
        contributions[name] = {
            "prediction": "REAL" if p >= REAL_THRESHOLD else "FAKE",
            "confidence": max(p, 1.0 - p),
        }
    return contributions


def _impact(value: float, high: float, medium: float | None) -> str:
    if high is not None and value >= high:
        return "high"
    if medium is not None and value >= medium:
        return "medium"
    return "low"


def stylometric_highlights(features: Mapping[str, float]) -> list[dict[str, Any]]:
    """Key stylometric values for the breakdown panel, with qualitative impact."""
    highlights: list[dict[str, Any]] = []
    for name, high, medium in _stylometric_highlights:
        value = float(features.get(name, 0.0))
        highlights.append(
            {
                "feature": name,
                "value": round(value, 4),
                "impact": _impact(value, high, medium),
            }
        )
    return highlights


def linguistic_flags(text: str) -> list[str]:
    """Plain-English warnings about stylistic red flags."""
    features = extract_all_stylometric_features(text)
    flags: list[str] = []

    caps_ratio = features.get("uppercase_char_ratio", 0.0)
    if caps_ratio >= HIGH_CAPS_RATIO:
        flags.append(f"Excessive capitalization ({caps_ratio:.0%} of letters)")

    all_caps_words = int(features.get("all_caps_word_count", 0.0))
    if all_caps_words >= MIN_ALL_CAPS_WORDS:
        flags.append(f"{all_caps_words} all-caps 'shouting' words")

    exclamation = features.get("exclamation_density", 0.0)
    if exclamation >= HIGH_EXCLAMATION_DENSITY:
        flags.append(f"Heavy use of exclamation marks ({exclamation:.3f} per character)")

    word_count = features.get("word_count", 0.0)
    quotation_count = round(features.get("quotation_density", 0.0) * features.get("char_count", 0.0))
    if quotation_count == 0 and word_count >= MIN_WORDS_FOR_QUOTATION_FLAG:
        flags.append("No direct quotations found in the article")

    subjectivity = features.get("subjectivity", 0.0)
    if subjectivity >= HIGH_SUBJECTIVITY:
        flags.append(f"Emotionally loaded language (subjectivity {subjectivity:.2f})")

    return flags


def top_token_features(
    document: str,
    linear_model: Any | None,
    vectorizer: Any | None,
    top_k: int = DEFAULT_TOP_K_TOKENS,
) -> list[dict[str, Any]]:
    """Features the logistic regression pushes hardest on this document."""
    if linear_model is None or vectorizer is None or not hasattr(linear_model, "coef_"):
        return []

    matrix = vectorizer.transform([document])
    row = matrix.toarray()[0] if hasattr(matrix, "toarray") else np.asarray(matrix)[0]
    weights = np.asarray(linear_model.coef_).ravel() * row
    if weights.size == 0:
        return []

    if hasattr(vectorizer, "feature_names_"):
        names = list(vectorizer.feature_names_)
    else:
        names = list(vectorizer.get_feature_names_out())

    magnitudes = np.abs(weights)
    order = np.argsort(magnitudes)[::-1]
    top = [(int(index), weights[index]) for index in order[: top_k] if magnitudes[index] > 0]
    if not top:
        return []

    strongest = magnitudes[top[0][0]]
    contributions = []
    for index, weight in top:
        contributions.append(
            {
                "feature": f"tfidf:{names[index]}",
                "value": round(float(weight), 6),
                "impact": _impact(
                    abs(weight),
                    0.75 * strongest,
                    0.4 * strongest,
                ),
            }
        )
    return contributions


def build_explanation(
    text: str,
    prediction: PredictionResult,
    artifacts: EnsembleArtifacts,
) -> dict[str, Any]:
    """Assemble the full explanation payload for an analyzed article."""
    document = _shared_preprocessor.clean(text)
    top_features = top_token_features(
        document,
        artifacts.classical.get("logistic_regression"),
        artifacts.vectorizer,
    )
    features = extract_all_stylometric_features(text)
    top_features.extend(stylometric_highlights(features))

    return {
        "top_features": top_features,
        "linguistic_flags": linguistic_flags(text),
        "model_contributions": model_contributions(prediction),
    }
