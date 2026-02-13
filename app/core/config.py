from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), str(PROJECT_ROOT / ".env.local")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LOG_LEVEL: str = "INFO"
    ORCHESTRATOR_PORT: int = 8001
    RAG_ENGINE_LOCAL_URL: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("RAG_ENGINE_LOCAL_URL", "RAG_ENGINE_URL", "RAG_SERVICE_URL"),
    )
    RAG_ENGINE_DOCKER_URL: str = "http://localhost:8000"
    RAG_ENGINE_HEALTH_PATH: str = "/health"
    RAG_ENGINE_PROBE_TIMEOUT_MS: int = 300
    RAG_ENGINE_BACKEND_TTL_SECONDS: int = 20
    RAG_ENGINE_FORCE_BACKEND: str | None = None

    QA_LITERAL_SEMANTIC_FALLBACK_ENABLED: bool = True
    QA_LITERAL_SEMANTIC_MIN_KEYWORD_OVERLAP: int = 2
    QA_LITERAL_SEMANTIC_MIN_SIMILARITY: float = 0.3

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"


settings = Settings()
