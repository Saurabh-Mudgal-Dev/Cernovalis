"""Tests for FastAPI application configuration and lifecycle."""

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import app


def test_settings_read_environment_values(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Test API")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("CORS_ORIGINS", '["https://ui.example"]')

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test API"
    assert settings.debug is True
    assert settings.cors_origins == ["https://ui.example"]


def test_health_and_lifespan(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main._load_predictor",
        lambda settings: (None, []),
    )
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert app.state.settings.app_name == app.title


def test_cors_preflight_for_frontend_origin(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main._load_predictor",
        lambda settings: (None, []),
    )
    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
