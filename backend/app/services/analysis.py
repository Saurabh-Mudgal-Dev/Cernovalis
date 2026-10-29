"""Shared analysis orchestration for the analyze and batch endpoints."""

from backend.app.schemas import (
    AnalysisExplanation,
    AnalyzeResponse,
    InputType,
    SourceCredibility,
    Verdict,
)
from backend.app.services.explainer import build_explanation, risk_level_for

PredictorLike = object  # PredictorService; duck-typed so tests can stub it


def analyze_text(predictor: PredictorLike, text: str, title: str | None = None) -> AnalyzeResponse:
    """Classify one article and build the full AnalyzeResponse payload."""
    prediction = predictor.predict(title=title, text=text)

    explanation = build_explanation(text, prediction, predictor.artifacts)
    return AnalyzeResponse(
        input_type=InputType.text,
        original_input=text or (title or ""),
        verdict=Verdict(prediction.verdict),
        confidence=prediction.confidence,
        risk_level=risk_level_for(prediction.verdict, prediction.confidence),
        source_credibility=SourceCredibility(),
        explanation=AnalysisExplanation(**explanation),
    )
