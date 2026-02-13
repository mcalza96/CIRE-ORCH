from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


QueryMode = Literal["literal_lista", "literal_normativa", "explicativa", "comparativa", "ambigua_scope"]


@dataclass(frozen=True)
class QueryIntent:
    mode: QueryMode
    rationale: str = ""


@dataclass(frozen=True)
class RetrievalPlan:
    mode: QueryMode
    chunk_k: int
    chunk_fetch_k: int
    summary_k: int
    require_literal_evidence: bool = False
    requested_standards: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    content: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerDraft:
    text: str
    mode: QueryMode
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClarificationRequest:
    question: str
    options: tuple[str, ...] = ()
