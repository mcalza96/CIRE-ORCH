from __future__ import annotations

import re
from typing import Any

from app.cartridges.models import AgentProfile
from app.agent.models import QueryIntent, QueryMode, RetrievalPlan

from app.core.config import settings


LITERAL_LIST_HINTS = ("lista", "enumera", "listado", "vinetas")

LITERAL_NORMATIVE_HINTS = (
    "texto exacto",
    "literal",
    "que exige",
    "requisito",
    "obligatorio",
)

COMPARATIVE_HINTS = ("compar", "difer", "vs", "entre", "respecto")

# Markers that suggest the user wants interpretation/impacts, not literal extraction.
INTERPRETIVE_HINTS = (
    "implica",
    "impacto",
    "analiza",
    "analice",
    "causa",
    "por que",
    "relaciona",
)

SCOPE_HINTS: dict[str, tuple[str, ...]] = {}

CONFLICT_MARKERS = (
    "conflicto",
    "represalia",
    "confidencial",
    "denuncia",
    "anonim",
    "proteccion de datos",
    "protección de datos",
    "rrhh",
    "se niega",
)

EVIDENCE_MARKERS = (
    "evidencia",
    "trazabilidad",
    "verificar",
    "registros",
    "informacion documentada",
    "información documentada",
)


def _router_hints(
    profile: AgentProfile | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    tuple[str, ...],
]:
    if profile is None:
        return (
            LITERAL_LIST_HINTS,
            LITERAL_NORMATIVE_HINTS,
            COMPARATIVE_HINTS,
            INTERPRETIVE_HINTS,
            SCOPE_HINTS,
            CONFLICT_MARKERS,
            EVIDENCE_MARKERS,
        )

    router = profile.router
    scope_hints: dict[str, tuple[str, ...]] = {
        key: tuple(values)
        for key, values in router.scope_hints.items()
        if isinstance(key, str) and isinstance(values, list)
    }

    return (
        tuple(router.literal_list_hints) or LITERAL_LIST_HINTS,
        tuple(router.literal_normative_hints) or LITERAL_NORMATIVE_HINTS,
        tuple(router.comparative_hints) or COMPARATIVE_HINTS,
        tuple(router.interpretive_hints) or INTERPRETIVE_HINTS,
        scope_hints or SCOPE_HINTS,
        tuple(router.conflict_markers) or CONFLICT_MARKERS,
        tuple(router.evidence_markers) or EVIDENCE_MARKERS,
    )


def extract_requested_scopes(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    """Extract standard names/codes from query using profile patterns or generic fallback."""
    text = (query or "").strip()
    lower_text = text.lower()
    found: set[str] = set()

    patterns = profile.router.scope_patterns if profile and profile.router.scope_patterns else []

    # If we have a profile with explicit patterns, use them.
    if patterns:
        for p in patterns:
            # Handle both ScopePattern objects and raw strings
            regex = p.regex if hasattr(p, "regex") else str(p)
            label = p.label if hasattr(p, "label") else None
            
            try:
                if re.search(regex, text, flags=re.IGNORECASE):
                    if label:
                        found.add(label.strip())
                    else:
                        matches = re.findall(regex, lower_text)
                        for m in matches:
                            if isinstance(m, tuple):
                                m = " ".join(filter(None, m))
                            found.add(str(m).strip().upper())
            except re.error:
                continue

    # Fallback to hints and entities
    if profile:
        for scope_label, hints in profile.router.scope_hints.items():
            if any(h.lower() in lower_text for h in hints) and scope_label not in found:
                found.add(scope_label.strip())
        
        for entity in profile.domain_entities:
            if len(entity) >= 4 and entity.lower() in lower_text and entity not in found:
                found.add(entity.strip())

    # Agnostic fallback for legacy/common standards
    if not found and not patterns:
        legacy_patterns = [r"\biso\s*[-:]?\s*(\d{4,5})\b", r"\b(9001|14001|45001)\b"]
        for lp in legacy_patterns:
            matches = re.findall(lp, lower_text)
            for m in matches:
                if isinstance(m, tuple):
                    m = " ".join(filter(None, m))
                val = str(m).strip().upper()
                if val.isdigit() and len(val) >= 4:
                    found.add(f"ISO {val}")
                else:
                    found.add(val)

    return tuple(sorted(list(found)))


def extract_requested_standards(query: str, profile: AgentProfile | None = None) -> set[str]:
    return set(extract_requested_scopes(query, profile=profile))


def has_clause_reference(query: str, profile: AgentProfile | None = None) -> bool:
    """Detect if query mentions a specific clause/article using profile patterns."""
    text = query.lower()
    patterns = profile.router.reference_patterns if profile else [r"\b\d+(?:\.\d+)+\b"]

    for p in patterns:
        if re.search(p, text):
            return True
    return False


def suggest_scope_candidates(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    _, _, _, _, scope_hints, _, _ = _router_hints(profile)
    text = (query or "").strip().lower()
    ranked: list[str] = []

    # First, use the explicit scope hints from the profile
    for standard, hints in scope_hints.items():
        if any(h in text for h in hints):
            ranked.append(standard)

    # Then, use the same pattern-based extraction as extract_requested_standards
    # to find additional candidates that might not have explicit hints but match patterns.
    pattern_based_candidates = extract_requested_standards(query, profile=profile)
    for candidate in pattern_based_candidates:
        if candidate not in ranked: # Avoid duplicates if already added by hints
            ranked.append(candidate)

    if ranked:
        seen: set[str] = set()
        ordered = [s for s in ranked if not (s in seen or seen.add(s))]
        return tuple(ordered)

    # If no candidates found by hints or patterns, return all available scope keys from hints
    if scope_hints:
        return tuple(scope_hints.keys())
    return tuple()


def detect_scope_candidates(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    _, _, _, _, scope_hints, _, _ = _router_hints(profile)
    requested = list(extract_requested_scopes(query, profile=profile))
    text = (query or "").strip().lower()
    for standard, hints in scope_hints.items():
        if standard in requested:
            continue
        if any(h in text for h in hints):
            requested.append(standard)
    return tuple(requested)


def detect_conflict_objectives(query: str, profile: AgentProfile | None = None) -> bool:
    _, _, _, _, _, conflict_markers, evidence_markers = _router_hints(profile)
    text = (query or "").strip().lower()
    has_conflict = any(marker in text for marker in conflict_markers)
    has_evidence = any(marker in text for marker in evidence_markers)
    return has_conflict and has_evidence


def classify_intent(query: str, profile: AgentProfile | None = None) -> QueryIntent:
    return classify_intent_with_trace(query, profile=profile)[0]


def classify_intent_with_trace(
    query: str,
    profile: AgentProfile | None = None,
) -> tuple[QueryIntent, dict[str, Any]]:
    """Return intent plus a trace payload suitable for observability."""

    text = (query or "").strip().lower()
    requested_standards = extract_requested_scopes(query, profile=profile)
    (
        literal_list_hints,
        literal_normative_hints,
        comparative_hints,
        interpretive_hints,
        _,
        _,
        _,
    ) = _router_hints(profile)

    # Backward-compatible fast path (kept for now) when v2 is disabled.
    if not bool(settings.ORCH_MODE_CLASSIFIER_V2):
        if len(requested_standards) >= 2 and any(k in text for k in interpretive_hints):
            return QueryIntent(
                mode="comparativa", rationale="multi-standard interpretive cross-impact"
            ), {
                "version": "v1",
                "mode": "comparativa",
                "confidence": 0.7,
                "reasons": ["heuristic:multi_scope+interpretive"],
            }
        if any(h in text for h in literal_list_hints):
            return QueryIntent(mode="literal_lista", rationale="list-like normative query"), {
                "version": "v1",
                "mode": "literal_lista",
                "confidence": 0.7,
                "reasons": ["heuristic:list"],
            }
        if any(h in text for h in literal_normative_hints):
            if any(h in text for h in interpretive_hints):
                mode: QueryMode = "comparativa" if len(requested_standards) >= 2 else "explicativa"
                return QueryIntent(mode=mode, rationale="interpretive question with clause refs"), {
                    "version": "v1",
                    "mode": mode,
                    "confidence": 0.65,
                    "reasons": ["heuristic:interpretive+clause"],
                }
            if has_clause_reference(query) and not requested_standards:
                return QueryIntent(
                    mode="ambigua_scope",
                    rationale="clause reference without explicit standard scope",
                ), {
                    "version": "v1",
                    "mode": "ambigua_scope",
                    "confidence": 0.7,
                    "reasons": ["heuristic:clause_without_scope"],
                }
            return QueryIntent(mode="literal_normativa", rationale="normative exactness query"), {
                "version": "v1",
                "mode": "literal_normativa",
                "confidence": 0.7,
                "reasons": ["heuristic:literal"],
            }
        if any(h in text for h in comparative_hints):
            return QueryIntent(mode="comparativa", rationale="cross-scope comparison"), {
                "version": "v1",
                "mode": "comparativa",
                "confidence": 0.6,
                "reasons": ["heuristic:comparative"],
            }
        return QueryIntent(mode="explicativa", rationale="general explanatory query"), {
            "version": "v1",
            "mode": "explicativa",
            "confidence": 0.55,
            "reasons": ["heuristic:default"],
        }

    from app.agent.mode_classifier import ModeClassification, classify_mode_v2

    classification: ModeClassification = classify_mode_v2(query, profile=profile)

    # Preserve an important v1 behavior: clause refs without explicit scope => ask for scope.
    # Only do this when the classifier is not strongly analytical/comparative.
    f = classification.features or {}
    if f.get("clause_refs_count", 0) >= 1 and f.get("requested_scopes_count", 0) == 0:
        if not bool(f.get("has_analysis_verb")) and not bool(f.get("has_comparative_marker")):
            return QueryIntent(
                mode="ambigua_scope", rationale="clause reference without explicit standard scope"
            ), {
                "version": "v2",
                "mode": "ambigua_scope",
                "confidence": 0.7,
                "reasons": ["guardrail:clause_without_scope"],
                "features": f,
                "blocked_modes": list(classification.blocked_modes),
            }

    intent = QueryIntent(
        mode=classification.mode,
        rationale=f"v2 confidence={round(classification.confidence, 2)} reasons={','.join(classification.reasons[:6])}",
    )
    trace = {
        "version": "v2",
        "mode": classification.mode,
        "confidence": float(classification.confidence),
        "reasons": list(classification.reasons),
        "features": f,
        "blocked_modes": list(classification.blocked_modes),
    }
    return intent, trace


def build_retrieval_plan(
    intent: QueryIntent,
    query: str = "",
    profile: AgentProfile | None = None,
) -> RetrievalPlan:
    requested_standards = extract_requested_scopes(query, profile=profile)
    mode_cfg = profile.retrieval.by_mode.get(intent.mode) if profile is not None else None

    if mode_cfg is not None:
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=int(mode_cfg.chunk_k),
            chunk_fetch_k=int(mode_cfg.chunk_fetch_k),
            summary_k=int(mode_cfg.summary_k),
            require_literal_evidence=bool(mode_cfg.require_literal_evidence),
            requested_standards=requested_standards,
        )

    if intent.mode in {"literal_lista", "literal_normativa"}:
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=45,
            chunk_fetch_k=220,
            summary_k=3,
            require_literal_evidence=True,
            requested_standards=requested_standards,
        )
    if intent.mode == "comparativa":
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=35,
            chunk_fetch_k=140,
            summary_k=5,
            # Comparativa is typically interpretive; still grounded in retrieved context,
            # but do not force clause-by-clause literal quoting.
            require_literal_evidence=False,
            requested_standards=requested_standards,
        )
    if intent.mode == "ambigua_scope":
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=0,
            chunk_fetch_k=0,
            summary_k=0,
            require_literal_evidence=True,
            requested_standards=requested_standards,
        )
    return RetrievalPlan(
        mode=intent.mode,
        chunk_k=30,
        chunk_fetch_k=120,
        summary_k=5,
        require_literal_evidence=False,
        requested_standards=requested_standards,
    )
