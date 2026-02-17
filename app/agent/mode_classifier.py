from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from app.cartridges.models import AgentProfile
from app.agent.models import QueryMode
from app.agent.policies import extract_requested_scopes
from app.agent.retrieval_planner import extract_clause_refs


@dataclass(frozen=True)
class ModeClassification:
    mode: QueryMode
    confidence: float
    reasons: tuple[str, ...] = ()
    features: dict[str, Any] | None = None
    blocked_modes: tuple[QueryMode, ...] = ()


_LITERAL_VERBS = (
    "cita",
    "transcribe",
    "texto exacto",
    "literal",
    "segun el texto",
    "según el texto",
    "que dice",
    "qué dice",
)

_ANALYSIS_VERBS = (
    "analice",
    "analiza",
    "impacta",
    "impacto",
    "impide",
    "causa",
    "por que",
    "por qué",
    "basandose",
    "basándose",
    "relaciona",
    "interrelacion",
    "interrelación",
)

_COMPARATIVE_MARKERS = (
    "vs",
    "compar",
    "difer",
    "entre",
    "ambas",
    "las tres",
)

_LIST_MARKERS = (
    "enumera",
    "lista",
    "listado",
    "viñetas",
    "entradas",
    "salidas",
)


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    t = _norm(text)
    return any(n in t for n in needles)


def extract_features(query: str, profile: AgentProfile | None = None) -> dict[str, Any]:
    text = _norm(query)
    clause_refs = extract_clause_refs(query, profile=profile)
    requested = extract_requested_scopes(query, profile=profile)
    sentence_count = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))

    list_markers = (
        tuple(profile.router.literal_list_hints)
        if profile is not None and profile.router.literal_list_hints
        else _LIST_MARKERS
    )
    comparative_markers = (
        tuple(profile.router.comparative_hints)
        if profile is not None and profile.router.comparative_hints
        else _COMPARATIVE_MARKERS
    )
    interpretive_markers = (
        tuple(profile.router.interpretive_hints)
        if profile is not None and profile.router.interpretive_hints
        else _ANALYSIS_VERBS
    )
    literal_markers = (
        tuple(profile.router.literal_normative_hints)
        if profile is not None and profile.router.literal_normative_hints
        else _LITERAL_VERBS
    )

    features: dict[str, Any] = {
        "query_length": len(text),
        "sentence_count": sentence_count,
        "clause_refs_count": len(clause_refs),
        "requested_scopes_count": len(requested),
        "has_literal_verb": _has_any(text, literal_markers),
        "has_analysis_verb": _has_any(text, interpretive_markers),
        "has_comparative_marker": _has_any(text, comparative_markers),
        "has_list_marker": _has_any(text, list_markers),
        "multi_scope": len(requested) >= 2,
        "multi_clause": len(clause_refs) >= 2,
        "clause_refs": clause_refs[:10],
        "requested_scopes": list(requested),
    }

    # A crude indicator of multi-objective analytical prompts.
    connectors = (" y ", " o ", " bas", " impact", " impid", " basado")
    features["multi_objective"] = bool(
        features["multi_clause"] and any(c in text for c in connectors)
    )
    return features


def classify_mode_v2(query: str, profile: AgentProfile | None = None) -> ModeClassification:
    f = extract_features(query, profile=profile)
    reasons: list[str] = []
    blocked: list[QueryMode] = []

    # Guardrails: never default to literal_normativa for analytical multi-clause prompts.
    if f.get("multi_clause") and f.get("has_analysis_verb"):
        blocked.append("literal_normativa")
        blocked.append("literal_lista")
        reasons.append("guardrail:block_literal_for_multiclause_analysis")

    literal_score = 0.0
    comparative_score = 0.0
    explanatory_score = 0.0
    list_score = 0.0

    if f.get("has_list_marker"):
        list_score += 2.0
        reasons.append("feature:list_marker")

    if f.get("has_literal_verb"):
        literal_score += 2.0
        reasons.append("feature:literal_verb")

    if f.get("clause_refs_count", 0) >= 1:
        literal_score += 1.0
        reasons.append("feature:clause_reference")

    if f.get("has_analysis_verb"):
        explanatory_score += 2.0
        comparative_score += 0.5
        reasons.append("feature:analysis_verb")

    if f.get("has_comparative_marker"):
        comparative_score += 2.0
        reasons.append("feature:comparative_marker")

    if f.get("multi_scope"):
        comparative_score += 3.2
        explanatory_score += 0.5
        reasons.append("feature:multi_scope")

    if f.get("multi_clause"):
        explanatory_score += 1.5
        comparative_score += 0.5
        reasons.append("feature:multi_clause")

    # Penalize literal when the prompt is clearly analytical.
    if f.get("multi_objective") and f.get("has_analysis_verb"):
        literal_score -= 2.0
        reasons.append("penalty:literal_for_multiobjective")

    candidates: list[tuple[QueryMode, float]] = [
        (cast(QueryMode, "literal_lista"), list_score),
        (cast(QueryMode, "literal_normativa"), literal_score),
        (cast(QueryMode, "comparativa"), comparative_score),
        (cast(QueryMode, "explicativa"), explanatory_score),
    ]

    # Apply hard blocks.
    filtered = [(m, s) for (m, s) in candidates if m not in blocked]
    if not filtered:
        filtered = candidates

    # Pick max; for ties prefer interpretive defaults over strict literal modes.
    tie_break_priority: dict[QueryMode, int] = {
        "explicativa": 0,
        "comparativa": 1,
        "literal_normativa": 2,
        "literal_lista": 3,
        "ambigua_scope": 4,
    }
    filtered.sort(key=lambda x: (-x[1], tie_break_priority.get(x[0], 99)))
    best_mode_raw, best_score = filtered[0]
    best_mode = cast(QueryMode, best_mode_raw)
    second_score = filtered[1][1] if len(filtered) > 1 else (best_score - 1.0)
    margin = best_score - second_score
    confidence = max(0.05, min(0.95, 0.5 + (margin / 4.0)))

    # If scores are low across the board, reduce confidence.
    if best_score <= 1.0:
        weak_signal = not any(
            [
                bool(f.get("has_literal_verb")),
                bool(f.get("has_list_marker")),
                bool(f.get("has_analysis_verb")),
                bool(f.get("has_comparative_marker")),
                int(f.get("clause_refs_count", 0)) >= 1,
                bool(f.get("multi_scope")),
            ]
        )
        if weak_signal and "explicativa" not in blocked:
            best_mode = cast(QueryMode, "explicativa")
            reasons.append("default:explicativa_for_low_signal")
        confidence = min(confidence, 0.55)
        reasons.append("low_signal")

    return ModeClassification(
        mode=best_mode,
        confidence=float(confidence),
        reasons=tuple(reasons[:16]),
        features=f,
        blocked_modes=tuple(blocked),
    )
