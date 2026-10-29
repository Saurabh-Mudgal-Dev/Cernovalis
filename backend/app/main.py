"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.app.api.routes.analyze import router as analyze_router
from backend.app.config import get_settings
from backend.app.services.predictor import ArtifactStatus, PredictorService, load_predictor


def _load_predictor(
    settings,
) -> tuple[PredictorService, list[ArtifactStatus]]:
    """Indirection so tests can stub artifact loading."""
    return load_predictor(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and release application-scoped resources."""
    settings = get_settings()
    app.state.settings = settings
    logger.info("Starting {}", settings.app_name)
    try:
        predictor, statuses = _load_predictor(settings)
        app.state.predictor = predictor
        app.state.model_status = statuses
        loaded = sum(1 for status in statuses if status.loaded)
        logger.info("Model artifacts loaded: {}/{}", loaded, len(statuses))
    except Exception as error:  # startup must not fail over a broken artifact
        logger.exception("Predictor could not be loaded: {}", error)
        app.state.predictor = None
        app.state.model_status = []
    try:
        yield
    finally:
        logger.info("Stopping {}", settings.app_name)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analyze_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, error: Exception
) -> JSONResponse:
    logger.exception("Unhandled error while serving {}", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
