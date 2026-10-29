"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Cernovalis API"
    app_version: str = "0.1.0"
    debug: bool = False
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    model_dir: Path = Path("backend/ml/models/saved")
    vectorizer_path: Path = Path("backend/ml/models/vectorizers/tfidf_vectorizer.joblib")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
