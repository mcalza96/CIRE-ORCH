from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

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

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"
    GEMINI_FLASH: str = "gemini-2.5-flash-lite"

    GROQ_API_KEY: str | None = None
    GROQ_MODEL_LIGHTWEIGHT: str = "openai/gpt-oss-20b"
    GROQ_MODEL_HEAVY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_DESIGN: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_CHAT: str = "openai/gpt-oss-20b"
    GROQ_MODEL_FORENSIC: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_ORCHESTRATION: str = "openai/gpt-oss-20b"
    GROQ_MODEL_SUMMARIZATION: str = "openai/gpt-oss-20b"

    RAG_SERVICE_SECRET: str | None = Field(default=None, validation_alias="RAG_SERVICE_SECRET")
    ORCH_AUTH_REQUIRED: bool = True
    SUPABASE_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
    )
    SUPABASE_ANON_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_ANON_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY"),
    )
    SUPABASE_JWKS_URL: str | None = None
    SUPABASE_JWT_AUDIENCE: str = "authenticated"
    SUPABASE_SERVICE_ROLE_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_SERVICE_ROLE_KEY", "SERVICE_ROLE_KEY"),
    )
    SUPABASE_MEMBERSHIPS_TABLE: str = "tenant_memberships"
    SUPABASE_MEMBERSHIP_USER_COLUMN: str = "user_id"
    SUPABASE_MEMBERSHIP_TENANT_COLUMN: str = "tenant_id"

    @property
    def resolved_supabase_jwks_url(self) -> str | None:
        if self.SUPABASE_JWKS_URL:
            return str(self.SUPABASE_JWKS_URL).strip()
        if self.SUPABASE_URL:
            return urljoin(str(self.SUPABASE_URL).rstrip("/") + "/", "auth/v1/.well-known/jwks.json")
        return None

    @property
    def resolved_supabase_rest_url(self) -> str | None:
        if not self.SUPABASE_URL:
            return None
        return urljoin(str(self.SUPABASE_URL).rstrip("/") + "/", "rest/v1")


settings = Settings()
