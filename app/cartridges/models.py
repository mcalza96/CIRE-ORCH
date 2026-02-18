from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScopePattern(BaseModel):
    label: str
    regex: str


ToolName = str


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
    complexity_patterns: list[str] = Field(default_factory=list)
    extraction_patterns: list[str] = Field(default_factory=list)
    calculation_patterns: list[str] = Field(default_factory=list)


class IntentRule(BaseModel):
    id: str
    mode: str
    all_keywords: list[str] = Field(default_factory=list)
    any_keywords: list[str] = Field(default_factory=list)
    all_patterns: list[str] = Field(default_factory=list)
    any_patterns: list[str] = Field(default_factory=list)
    all_markers: list[str] = Field(default_factory=list)
    any_markers: list[str] = Field(default_factory=list)


class QueryModeConfig(BaseModel):
    require_literal_evidence: bool = False
    allow_inference: bool = True
    retrieval_profile: str | None = None
    tool_hints: list[ToolName] = Field(default_factory=list)
    execution_plan: list[ToolName] = Field(default_factory=list)
    coverage_requirements: dict[str, Any] = Field(default_factory=dict)
    decomposition_policy: dict[str, Any] = Field(default_factory=dict)
    response_contract: str | None = None


class QueryModesPolicy(BaseModel):
    default_mode: str = "default"
    modes: dict[str, QueryModeConfig] = Field(default_factory=dict)
    intent_rules: list[IntentRule] = Field(default_factory=list)


class RetrievalModeConfig(BaseModel):
    chunk_k: int = Field(ge=0, le=120)
    chunk_fetch_k: int = Field(ge=0, le=500)
    summary_k: int = Field(ge=0, le=30)
    require_literal_evidence: bool = False


class SearchHint(BaseModel):
    term: str
    expand_to: list[str] = Field(default_factory=list)


class RetrievalPolicy(BaseModel):
    by_mode: dict[str, RetrievalModeConfig] = Field(default_factory=dict)
    search_hints: list[SearchHint] = Field(default_factory=list)
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)


class SynthesisPolicy(BaseModel):
    system_persona: str = ""
    citation_format: str = "C#/R#"
    strict_reference_label: str = "Reference"
    strict_subject_label: str = "Claim"
    strict_system_prompt_template: str = ""
    interpretive_system_prompt_template: str = ""
    strict_style_template: str = ""
    interpretive_style_template: str = ""
    identity_role_prefix: str = ""
    identity_tone_prefix: str = ""
    user_prompt_template: str = ""
    citation_schema_version: str = "v1"
    citation_required_fields: list[str] = Field(
        default_factory=lambda: ["id", "standard", "clause_id", "quote"]
    )
    citation_render_template: str = '{id} | {standard} | clausula {clause_id} | "{snippet}"'
    citation_sort_policy: str = "completeness_then_score"
    citation_noise_filters: list[str] = Field(default_factory=list)
    citation_repair_enabled: bool = True
    min_structured_citation_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    response_contracts: dict[str, list[str]] = Field(default_factory=dict)
    default_response_contract: list[str] = Field(
        default_factory=lambda: [
            "Hechos citados",
            "Inferencias",
            "Brechas",
            "Recomendaciones",
            "Confianza y supuestos",
        ]
    )
    grounded_inference_min_citations: int = Field(default=2, ge=1, le=8)
    unsupported_claim_label: str = "Hipotesis"
    confidence_scale: list[str] = Field(default_factory=lambda: ["alta", "media", "baja"])
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
    require_structured_sections: bool = False
    required_sections: list[str] = Field(default_factory=list)
    min_citations_per_inference: int = Field(default=2, ge=1, le=8)
    block_strong_claims_without_evidence: bool = True
    strong_claim_markers: list[str] = Field(
        default_factory=lambda: [
            "riesgo critico",
            "incumplimiento",
            "no conformidad mayor",
            "sancion",
            "multa",
        ]
    )


class ExpectationRule(BaseModel):
    id: str
    description: str
    scopes: list[str] = Field(default_factory=list)
    clause_refs: list[str] = Field(default_factory=list)
    applies_to_modes: list[str] = Field(default_factory=list)
    required_evidence_markers: list[str] = Field(default_factory=list)
    optional_evidence_markers: list[str] = Field(default_factory=list)
    missing_risk: str = "Riesgo operativo por ausencia de evidencia"
    severity: Literal["low", "medium", "high", "critical"] = "medium"


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
    expectations: list[ExpectationRule] = Field(default_factory=list)

    router: RouterHeuristics = Field(default_factory=RouterHeuristics)
    query_modes: QueryModesPolicy = Field(default_factory=QueryModesPolicy)
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
IntentRule.model_rebuild()
QueryModeConfig.model_rebuild()
QueryModesPolicy.model_rebuild()
SynthesisPolicy.model_rebuild()
CapabilitiesPolicy.model_rebuild()
ReasoningBudget.model_rebuild()
ToolPolicy.model_rebuild()
ExpectationRule.model_rebuild()
