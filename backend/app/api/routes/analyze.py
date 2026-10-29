"""POST /api/v1/analyze endpoint."""

from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
)
from backend.app.services.analysis import analyze_text

router = APIRouter(prefix="/api/v1", tags=["analysis"])


def _get_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded; check the models directory.",
        )
    return predictor


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    if payload.url is not None:
        raise HTTPException(
            status_code=501,
            detail="URL analysis is not implemented yet.",
        )

    predictor = _get_predictor(request)
    try:
        return analyze_text(predictor, payload.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
