from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScopePattern(BaseModel):
    label: str
    regex: str


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


class AgentProfile(BaseModel):
    profile_id: str
    version: str = "1.0.0"
    status: Literal["draft", "active"] = "active"

    domain_entities: list[str] = Field(default_factory=list)
    intent_examples: dict[str, list[str]] = Field(default_factory=dict)
    clarification_rules: list[dict[str, Any]] = Field(default_factory=list)

    router: RouterHeuristics = Field(default_factory=RouterHeuristics)
    retrieval: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    synthesis: SynthesisPolicy = Field(default_factory=SynthesisPolicy)


AgentProfile.model_rebuild()
RetrievalPolicy.model_rebuild()
RouterHeuristics.model_rebuild()
SynthesisPolicy.model_rebuild()
