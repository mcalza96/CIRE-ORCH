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

    # Retrieval contract selection:
    # - legacy: uses /api/v1/debug/retrieval/* endpoints
    # - advanced: uses /api/v1/retrieval/* contract endpoints (validate-scope/hybrid/multi-query/explain)
    ORCH_RETRIEVAL_CONTRACT: str = "advanced"
    ORCH_MULTIHOP_FALLBACK: bool = True
    ORCH_SEMANTIC_PLANNER: bool = False
    ORCH_PLANNER_MAX_QUERIES: int = 5
    ORCH_PLANNER_MODEL: str | None = None
    ORCH_DETERMINISTIC_SUBQUERY_SEMANTIC_TAIL: bool = True

    # Multi-query promotion/iteration (agentic kernel guardrails)
    ORCH_MULTI_QUERY_PRIMARY: bool = False
    ORCH_MULTI_QUERY_REFINE: bool = False
    ORCH_MULTI_QUERY_MIN_ITEMS: int = 6
    ORCH_MULTI_QUERY_FALLBACK_MAX_QUERIES: int = 3
    ORCH_MULTI_QUERY_EVALUATOR: bool = False
    EARLY_EXIT_COVERAGE_ENABLED: bool = True
    ORCH_EVALUATOR_MODEL: str | None = None

    # Agnostic coverage gate: ensure multi-scope queries retrieve evidence per requested scope.
    ORCH_COVERAGE_GATE_ENABLED: bool = True
    ORCH_COVERAGE_GATE_TOP_N: int = 12
    ORCH_COVERAGE_GATE_MAX_MISSING: int = 2
    ORCH_COVERAGE_GATE_STEP_BACK: bool = True
    ORCH_COVERAGE_REQUIRED: bool = True
    ORCH_COVERAGE_AUTO_PARTIAL_COMPARATIVA: bool = True
    ORCH_LITERAL_LOCK_ENABLED: bool = True
    ORCH_MIN_SCORE_BACKSTOP_ENABLED: bool = True
    ORCH_MIN_SCORE_BACKSTOP_TOP_N: int = 6
    ORCH_LITERAL_REF_MIN_COVERAGE_RATIO: float = 0.7
    STRICT_LITERAL_CLAUSE_VALIDATION_ONLY: bool = True
    COMPOUND_QUERY_SPLIT_ENABLED: bool = True
    COMPOUND_QUERY_MAX_PARTS: int = 3

    # Raptor summaries (optional). Advanced contract does not include summaries,
    # so we can call the debug summaries endpoint in a controlled way.
    ORCH_RAPTOR_SUMMARIES_ENABLED: bool = False

    # Mode classifier v2 (agnostic, feature-based) + optional LLM advisor.
    ORCH_MODE_CLASSIFIER_V2: bool = True
    ORCH_MODE_ADVISOR_ENABLED: bool = False
    ORCH_MODE_ADVISOR_MODEL: str | None = None
    ORCH_MODE_LOW_CONFIDENCE_THRESHOLD: float = 0.55

    # Level-4-ish internal retries driven by classifier + validation.
    ORCH_MODE_AUTORETRY_ENABLED: bool = True
    ORCH_MODE_AUTORETRY_MAX_ATTEMPTS: int = 2
    ORCH_GRAPH_MAX_RETRIES: int = 1
    ORCH_GRAPH_MIN_AVG_SCORE: float = 0.12

    # Human-in-the-loop fallback when confidence low and internal retry not enough.
    ORCH_MODE_HITL_ENABLED: bool = False

    # Cartridge system (Level 4 dynamic profile injection)
    ORCH_CARTRIDGES_DIR: str | None = None
    ORCH_DEFAULT_PROFILE_ID: str = "base"
    ORCH_TENANT_PROFILE_MAP: str | None = None
    ORCH_TENANT_PROFILE_WHITELIST: str | None = None
    ORCH_AGENT_PROFILE_HEADER: str = "X-Agent-Profile"
    ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED: bool = True
    ORCH_DEV_PROFILE_ASSIGNMENTS_FILE: str = ".state/tenant_profile_assignments.json"

    # Optional DB-backed cartridge override (tenant private profiles)
    ORCH_CARTRIDGE_DB_ENABLED: bool = False
    ORCH_CARTRIDGE_DB_TABLE: str = "tenant_configs"
    ORCH_CARTRIDGE_DB_TENANT_COLUMN: str = "tenant_id"
    ORCH_CARTRIDGE_DB_PROFILE_COLUMN: str = "agent_profile"
    ORCH_CARTRIDGE_DB_PROFILE_ID_COLUMN: str = "profile_id"
    ORCH_CARTRIDGE_DB_VERSION_COLUMN: str = "profile_version"
    ORCH_CARTRIDGE_DB_STATUS_COLUMN: str = "status"
    ORCH_CARTRIDGE_DB_UPDATED_COLUMN: str = "updated_at"
    ORCH_CARTRIDGE_DB_TIMEOUT_SECONDS: float = 1.8
    ORCH_CARTRIDGE_DB_CACHE_TTL_SECONDS: int = 60

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
            return urljoin(
                str(self.SUPABASE_URL).rstrip("/") + "/", "auth/v1/.well-known/jwks.json"
            )
        return None

    @property
    def resolved_supabase_rest_url(self) -> str | None:
        if not self.SUPABASE_URL:
            return None
        return urljoin(str(self.SUPABASE_URL).rstrip("/") + "/", "rest/v1")


settings = Settings()
