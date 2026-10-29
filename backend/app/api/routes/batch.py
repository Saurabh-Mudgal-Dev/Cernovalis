"""POST /api/v1/batch CSV upload endpoint."""

import csv
import io

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from backend.app.schemas import (
    AnalysisExplanation,
    AnalyzeResponse,
    BatchAnalysisItem,
    BatchAnalyzeResponse,
    InputType,
    SourceCredibility,
    Verdict,
)
from backend.app.services.explainer import build_explanation, risk_level_for

router = APIRouter(prefix="/api/v1", tags=["analysis"])

MAX_ROWS = 1000
REQUIRED_COLUMN = "text"


class BatchValidationError(ValueError):
    """Raised when the uploaded CSV cannot be processed at all."""


def parse_csv_rows(raw: bytes) -> list[dict[str, str]]:
    """Parse the CSV upload into prepared row dicts; raise BatchValidationError on bad input."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise BatchValidationError("The CSV file must be UTF-8 encoded.") from error

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
    if not fieldnames:
        raise BatchValidationError("The CSV file has no header row.")

    if REQUIRED_COLUMN not in fieldnames:
        raise BatchValidationError(
            f"The CSV must contain a '{REQUIRED_COLUMN}' column."
        )

    rows: list[dict[str, str]] = []
    for record in reader:
        cells = {
            (name or "").strip().lower(): (record.get(name) or "").strip()
            for name in (reader.fieldnames or [])
        }
        rows.append(
            {
                "title": cells.get("title", ""),
                "text": cells.get(REQUIRED_COLUMN, ""),
            }
        )
        if len(rows) >= MAX_ROWS:
            break

    extra = sum(1 for _ in reader)
    if extra:
        raise BatchValidationError(
            f"The CSV exceeds the {MAX_ROWS}-row limit ({extra} additional rows)."
        )
    return rows


def _analyze_row(predictor, row: dict[str, str], row_number: int) -> BatchAnalysisItem:
    """Analyze one CSV row; processing errors become per-row failures."""
    try:
        text = row["text"]
        title = row["title"]
        if not text and not title:
            raise ValueError("Row has no title and no article text.")

        prediction = predictor.predict(title=title or None, text=text or None)
        explanation = build_explanation(text or title, prediction, predictor.artifacts)
        result = AnalyzeResponse(
            input_type=InputType.text,
            original_input=text or title or "",
            verdict=Verdict(prediction.verdict),
            confidence=prediction.confidence,
            risk_level=risk_level_for(prediction.verdict, prediction.confidence),
            source_credibility=SourceCredibility(),
            explanation=AnalysisExplanation(**explanation),
        )
        return BatchAnalysisItem(row=row_number, result=result)
    except Exception as error:
        logger.info("Batch row {} failed: {}", row_number, error)
        return BatchAnalysisItem(row=row_number, error=str(error) or "Processing failed")


@router.post("/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(
    request: Request,
    file: UploadFile = File(..., description="CSV with 'text' and optional 'title' columns."),
) -> BatchAnalyzeResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded; check the models directory.",
        )

    try:
        rows = parse_csv_rows(await file.read())
    except BatchValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    items = [_analyze_row(predictor, row, number) for number, row in enumerate(rows, start=1)]
    succeeded = sum(1 for item in items if item.result is not None)
    batch = BatchAnalyzeResponse(
        total=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        results=items,
    )
    logger.info("Batch analysis: {} total, {} succeeded, {} failed", batch.total, batch.succeeded, batch.failed)
    return batch
