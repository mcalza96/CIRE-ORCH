from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScopePattern(BaseModel):
    label: str
    regex: str


ToolName = Literal[
    "semantic_retrieval",
    "structural_extraction",
    "logical_comparison",
    "python_calculator",
    "citation_validator",
]


class RouterHeuristics(BaseModel):
    literal_list_hints: list[str] = Field(default_factory=list)
    literal_normative_hints: list[str] = Field(default_factory=list)
    comparative_hints: list[str] = Field(default_factory=list)
    interpretive_hints: list[str] = Field(default_factory=list)
    conflict_markers: list[str] = Field(default_factory=list)
    evidence_markers: list[str] = Field(default_factory=list)
    scope_hints: dict[str, list[str]] = Field(default_factory=dict)
    scope_patterns: list[ScopePattern] = Field(default_factory=list)
    reference_patterns: list[str] = Field(default_factory=list)


class RetrievalModeConfig(BaseModel):
    chunk_k: int
    chunk_fetch_k: int
    summary_k: int
    require_literal_evidence: bool = False


class SearchHint(BaseModel):
    term: str
    expand_to: list[str] = Field(default_factory=list)


class RetrievalPolicy(BaseModel):
    by_mode: dict[str, RetrievalModeConfig] = Field(
        default_factory=lambda: {
            "literal_lista": RetrievalModeConfig(
                chunk_k=45,
                chunk_fetch_k=220,
                summary_k=3,
                require_literal_evidence=True,
            ),
            "literal_normativa": RetrievalModeConfig(
                chunk_k=45,
                chunk_fetch_k=220,
                summary_k=3,
                require_literal_evidence=True,
            ),
            "comparativa": RetrievalModeConfig(
                chunk_k=35,
                chunk_fetch_k=140,
                summary_k=5,
                require_literal_evidence=False,
            ),
            "ambigua_scope": RetrievalModeConfig(
                chunk_k=0,
                chunk_fetch_k=0,
                summary_k=0,
                require_literal_evidence=True,
            ),
            "explicativa": RetrievalModeConfig(
                chunk_k=30,
                chunk_fetch_k=120,
                summary_k=5,
                require_literal_evidence=False,
            ),
        }
    )
    search_hints: list[SearchHint] = Field(default_factory=list)
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)


class SynthesisPolicy(BaseModel):
    system_persona: str = (
        "Responde con evidencia del contexto recuperado y evita afirmaciones sin sustento."
    )
    citation_format: str = "C#/R#"
    strict_reference_label: str = "Referencia"
    strict_subject_label: str = "Afirmacion"
    scope_hints: dict[str, list[str]] = Field(default_factory=dict)
    scope_patterns: list[str] = Field(default_factory=list)
    reference_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b\d+(?:\.\d+)+\b",
        ]
    )
    clarification_rules: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_rules: list[str] = Field(
        default_factory=lambda: [
            "Cada afirmacion relevante debe referenciar evidencia recuperada.",
            "Si no hay evidencia suficiente, indicarlo explicitamente.",
            "No inventar referencias ni citas.",
        ]
    )
    strict_style: list[str] = Field(
        default_factory=lambda: [
            "Para cada afirmacion: requisito | cita breve | fuente.",
            "No inventar texto normativo.",
        ]
    )
    interpretive_style: list[str] = Field(
        default_factory=lambda: [
            "Puedes conectar evidencias separadas, pero transparenta inferencias.",
            "Incluye referencias al final de cada punto.",
        ]
    )


class ProfileMeta(BaseModel):
    id: str = "base_v1"
    description: str = ""
    owner: str = "orchestrator"


class IdentityPolicy(BaseModel):
    role: str = "Analista tecnico"
    tone: str = "Formal y basado en evidencia"
    style_guide: list[str] = Field(default_factory=list)


class ValidationPolicy(BaseModel):
    require_citations: bool = True
    forbidden_concepts: list[str] = Field(default_factory=list)
    fallback_message: str = "No tengo informacion suficiente en el contexto para responder."


class ReasoningBudget(BaseModel):
    max_steps: int = Field(default=4, ge=1, le=12)
    max_reflections: int = Field(default=2, ge=0, le=6)


class ToolPolicy(BaseModel):
    enabled: bool = True
    max_input_chars: int = Field(default=6000, ge=256, le=32000)
    max_output_chars: int = Field(default=12000, ge=256, le=64000)
    max_expression_chars: int = Field(default=256, ge=16, le=4096)
    timeout_ms: int = Field(default=200, ge=20, le=5000)
    max_operations: int = Field(default=1000, ge=10, le=100000)
    schema_hint: str | None = None


class CapabilitiesPolicy(BaseModel):
    reasoning_level: Literal["low", "high"] = "low"
    allowed_tools: list[ToolName] = Field(
        default_factory=lambda: ["semantic_retrieval", "citation_validator"]
    )
    reasoning_budget: ReasoningBudget = Field(default_factory=ReasoningBudget)
    tool_policies: dict[ToolName, ToolPolicy] = Field(default_factory=dict)


class ProfileResolution(BaseModel):
    source: Literal["db", "header", "dev_map", "tenant_map", "tenant_yaml", "base"]
    requested_profile_id: str | None = None
    applied_profile_id: str
    decision_reason: str


class AgentProfile(BaseModel):
    profile_id: str
    version: str = "1.0.0"
    status: Literal["draft", "active"] = "active"
    meta: ProfileMeta = Field(default_factory=ProfileMeta)
    identity: IdentityPolicy = Field(default_factory=IdentityPolicy)

    domain_entities: list[str] = Field(default_factory=list)
    intent_examples: dict[str, list[str]] = Field(default_factory=dict)
    clarification_rules: list[dict[str, Any]] = Field(default_factory=list)

    router: RouterHeuristics = Field(default_factory=RouterHeuristics)
    retrieval: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    validation: ValidationPolicy = Field(default_factory=ValidationPolicy)
    synthesis: SynthesisPolicy = Field(default_factory=SynthesisPolicy)
    capabilities: CapabilitiesPolicy = Field(default_factory=CapabilitiesPolicy)


class ResolvedAgentProfile(BaseModel):
    profile: AgentProfile
    resolution: ProfileResolution


AgentProfile.model_rebuild()
RetrievalPolicy.model_rebuild()
RouterHeuristics.model_rebuild()
SynthesisPolicy.model_rebuild()
CapabilitiesPolicy.model_rebuild()
ReasoningBudget.model_rebuild()
ToolPolicy.model_rebuild()
