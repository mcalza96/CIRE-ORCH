from __future__ import annotations

from typing import Any

from app.agent.models import ClarificationRequest, QueryIntent
from app.profiles.models import AgentProfile


def has_user_clarification_marker(query: str) -> bool:
    normalized_query = (query or "").lower()
    return (
        "aclaración de alcance:" in normalized_query
        or "aclaracion de alcance:" in normalized_query
        or "modo preferido en sesión:" in normalized_query
        or "modo preferido en sesion:" in normalized_query
        or "__clarified_scope__=true" in normalized_query
    )


def coverage_preference(query: str) -> str:
    lowered = (query or "").lower()
    if "__coverage__=partial" in lowered:
        return "partial"
    if "__coverage__=full" in lowered:
        return "full"
    partial_markers = (
        "aceptar respuesta parcial",
        "respuesta parcial",
        "parcial",
    )
    full_markers = (
        "cobertura completa",
        "exigir cobertura completa",
        "completa",
    )
    has_scope_clarification = (
        "__clarified_scope__=true" in lowered
        or "aclaracion de alcance:" in lowered
        or "aclaración de alcance:" in lowered
    )
    if not has_scope_clarification and not any(
        marker in lowered for marker in [*partial_markers, *full_markers]
    ):
        return "unspecified"
    if any(marker in lowered for marker in partial_markers):
        return "partial"
    if any(marker in lowered for marker in full_markers):
        return "full"
    return "unspecified"


def build_profile_clarification(
    *,
    profile: AgentProfile | None,
    has_user_clarification: bool,
    detected_scopes: list[str],
    intent: QueryIntent,
    intent_trace: dict[str, Any],
    normalized_query: str,
    low_confidence_threshold: float,
) -> ClarificationRequest | None:
    if profile is None or has_user_clarification:
        return None

    scopes_text = ", ".join(detected_scopes)
    for rule in profile.clarification_rules:
        if not isinstance(rule, dict):
            continue

        min_scope_raw: Any = rule.get("min_scope_count")
        if min_scope_raw is not None:
            try:
                min_scope_count = int(min_scope_raw)
                if len(detected_scopes) < min_scope_count:
                    continue
            except (ValueError, TypeError):
                pass

        target_mode = rule.get("mode")
        if target_mode and target_mode != intent.mode:
            continue

        all_markers = [str(m).lower() for m in rule.get("all_markers", [])]
        any_markers = [str(m).lower() for m in rule.get("any_markers", [])]

        virtual_markers = {f"__mode__={intent.mode}"}
        confidence = float(intent_trace.get("confidence") or 1.0)
        if confidence < float(low_confidence_threshold or 0.55):
            virtual_markers.add("__low_confidence__")

        def _match_marker(m: str) -> bool:
            return m in normalized_query or m in virtual_markers

        if all_markers and not all(_match_marker(m) for m in all_markers):
            continue
        if any_markers and not any(_match_marker(m) for m in any_markers):
            continue

        template = str(rule.get("question_template") or rule.get("question") or "").strip()
        if not template:
            continue
        question = template.format(scopes=scopes_text)
        options = tuple(str(o).strip() for o in rule.get("options", []))
        return ClarificationRequest(question=question, options=options)
    return None
