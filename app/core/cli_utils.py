from __future__ import annotations

import re
import httpx


def _current_clarification_round(query: str) -> int:
    match = re.search(r"__clarification_round__=(\d+)", str(query or ""), flags=re.IGNORECASE)
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _extract_scope_list_from_answer(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[,;|]", raw)
    scopes: list[str] = []
    for part in parts:
        value = str(part or "").strip()
        if not value or len(value) < 3:
            continue
        normalized = value.lower()
        if normalized in {"comparar multiples", "alcance unico", "respuesta parcial"}:
            continue
        # Keep only scope-like values (codes/standards), not generic phrases.
        if not (
            re.search(r"\d", value)
            or re.search(
                r"\b(?:iso|iec|nom|nmx|nfpa|osha|en|une|iram|bs|din)\b", value, flags=re.IGNORECASE
            )
        ):
            continue
        scopes.append(value)
    return scopes[:5]


def looks_like_scope_answer(text: str) -> bool:
    return len(_extract_scope_list_from_answer(text)) > 0


def propose_scope_candidates(query: str, clarification_answer: str) -> list[str]:
    hay = f"{str(query or '')} {str(clarification_answer or '')}".lower()
    digits = re.findall(r"\b(\d{4,5})\b", hay)
    unique_digits: list[str] = []
    for d in digits:
        if d not in unique_digits:
            unique_digits.append(d)
    if unique_digits:
        return [f"ISO {d}" for d in unique_digits[:5]]
    if "iso" in hay:
        # Conservative default for ISO-only comparative prompts.
        return ["ISO 9001", "ISO 14001", "ISO 45001"]
    return []


def rewrite_query_with_clarification(
    original_query: str,
    clarification_answer: str,
    clarification_kind: str | None = None,
) -> str:
    text = (clarification_answer or "").strip()
    if not text:
        return original_query
    lowered = text.lower().strip()
    kind = str(clarification_kind or "clarification").strip().lower()
    round_no = _current_clarification_round(original_query) + 1
    base_tags = f"__clarification_round__={round_no} "

    plan_tag = ""
    if kind == "plan_approval":
        if lowered in {"si", "sí", "ok", "aprobado", "confirmo", "confirmado"}:
            plan_tag = "__plan_approved__=true "
        elif lowered not in {"no", "ajustar", "cambiar alcance"}:
            safe_feedback = re.sub(r"[^a-z0-9_\- ]+", "", lowered)[:80].strip().replace(" ", "_")
            if safe_feedback:
                plan_tag = f"__plan_feedback__={safe_feedback} "

    choice_tag = ""
    confirm_tag = ""
    if kind == "clarification":
        if lowered in {"si", "sí", "ok", "confirmo", "confirmado", "si, continuar"}:
            confirm_tag = "__clarification_confirmed__=true "
        if lowered == "comparar multiples":
            choice_tag = "__clarification_choice__=compare_multiple "
        elif lowered == "alcance unico":
            choice_tag = "__clarification_choice__=single_scope "
        elif lowered == "respuesta parcial":
            choice_tag = "__clarification_choice__=partial_response "

    scopes = _extract_scope_list_from_answer(text)
    scopes_tag = f"__requested_scopes__=[{'|'.join(scopes)}] " if scopes else ""

    if re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", lowered):
        return (
            f"{original_query}\n\n{base_tags}{plan_tag}{choice_tag}{scopes_tag}"
            f"{confirm_tag}"
            f"__clarified_mode__={lowered} Aclaracion de modo: {text}."
        )

    coverage_tag = ""
    if lowered in {"respuesta parcial", "aceptar respuesta parcial"}:
        coverage_tag = "__coverage__=partial "
    elif lowered in {"cobertura completa", "exigir cobertura completa"}:
        coverage_tag = "__coverage__=full "
    return (
        f"{original_query}\n\n{base_tags}{plan_tag}{choice_tag}{scopes_tag}"
        f"{confirm_tag}"
        f"__clarified_scope__=true {coverage_tag}Aclaracion de alcance: {text}."
    )


def apply_mode_override(query: str, forced_mode: str | None) -> str:
    mode = str(forced_mode or "").strip()
    if not mode:
        return query
    lowered = mode.lower()
    if not re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", lowered):
        return query
    return f"__mode__={lowered} {query}".strip()


def prompt(message: str) -> str:
    return input(message).strip()


def short_token(token: str) -> str:
    if len(token) < 12:
        return token
    return f"{token[:6]}...{token[-4:]}"


def require_orch_health(orchestrator_url: str) -> None:
    base = orchestrator_url.rstrip("/")
    health_url = f"{base}/health"
    try:
        response = httpx.get(health_url, timeout=3.0)
    except Exception as exc:
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {health_url}\n💡 Ejecuta ./stack.sh up"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {health_url}\n💡 Ejecuta ./stack.sh up"
        )
