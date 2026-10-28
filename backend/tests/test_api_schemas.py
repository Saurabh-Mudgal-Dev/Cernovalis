"""Tests for API request and response schema validation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    AnalysisExplanation,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalysisItem,
    BatchAnalyzeResponse,
    CredibilityLabel,
    InputType,
    RiskLevel,
    SourceCredibility,
    Verdict,
)


def test_analyze_request_accepts_text_or_url():
    text_request = AnalyzeRequest(text="A news article")
    url_request = AnalyzeRequest(url="https://example.com/article")

    assert text_request.text == "A news article"
    assert text_request.url is None
    assert str(url_request.url) == "https://example.com/article"
    assert url_request.text is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": "article", "url": "https://example.com"},
        {"text": ""},
        {"url": "not-a-url"},
        {"text": "article", "unknown": "field"},
    ],
)
def test_analyze_request_rejects_invalid_inputs(payload):
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_analyze_response_validates_and_serializes_nested_data():
    response = AnalyzeResponse(
        input_type=InputType.text,
        original_input="Article text",
        extracted_text="Clean article text",
        verdict=Verdict.fake,
        confidence=0.87,
        risk_level=RiskLevel.high,
        source_credibility=SourceCredibility(
            domain="example.com",
            score=0.23,
            label=CredibilityLabel.unreliable,
            known_flags=["satire"],
        ),
        explanation=AnalysisExplanation(linguistic_flags=["High capitalization"]),
    )
    payload = response.model_dump(mode="json")

    assert response.timestamp.tzinfo == timezone.utc
    assert payload["verdict"] == "FAKE"
    assert payload["source_credibility"]["label"] == "UNRELIABLE"
    assert payload["explanation"]["linguistic_flags"] == ["High capitalization"]
    assert datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_analyze_response_rejects_confidence_out_of_range(confidence):
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            input_type="text",
            original_input="Article text",
            verdict="REAL",
            confidence=confidence,
            risk_level="LOW",
        )


def test_batch_response_requires_consistent_counts_and_item_outcome():
    success = BatchAnalysisItem(
        row=1,
        result=AnalyzeResponse(
            input_type="text",
            original_input="Article text",
            verdict="REAL",
            confidence=0.8,
            risk_level="LOW",
        ),
    )
    failure = BatchAnalysisItem(row=2, error="No article text")

    batch = BatchAnalyzeResponse(
        total=2,
        succeeded=1,
        failed=1,
        results=[success, failure],
    )

    assert batch.total == len(batch.results)
    with pytest.raises(ValidationError):
        BatchAnalyzeResponse(total=2, succeeded=2, failed=1, results=[success, failure])
    with pytest.raises(ValidationError):
        BatchAnalysisItem(row=3, error="failed", result=success.result)
