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
        compact = re.sub(r"\s+", " ", value).strip()
        # Keep only strict scope-like values (codes/standards), not generic phrases.
        prefix_with_code = bool(
            re.search(
                r"^(?:iso|iec|nom|nmx|nfpa|osha|en|une|iram|bs|din)\s*[-:_]?\s*\d{2,6}(?:[:\-]\d{4})?$",
                compact,
                flags=re.IGNORECASE,
            )
        )
        compact_code = bool(re.search(r"^[A-Za-z]{2,12}[-_ ]?\d{2,6}$", compact))
        numeric_code = bool(re.search(r"^\d{3,6}$", compact))
        # Sometimes users type '9001', which should be recognized!
        if not (prefix_with_code or compact_code or numeric_code):
            continue
        
        # If it's just a number, prepend ISO assuming that's the default context for such numbers
        if numeric_code and not prefix_with_code and not compact_code:
             compact = f"ISO {compact}"
             
        scopes.append(compact)
    return scopes[:5]


def looks_like_scope_answer(text: str) -> bool:
    return len(_extract_scope_list_from_answer(text)) > 0


def extract_scope_list_from_answer(text: str) -> list[str]:
    return _extract_scope_list_from_answer(text)


def build_clarification_context(
    *,
    clarification: dict[str, object] | None,
    answer_text: str,
    round_no: int,
) -> dict[str, object]:
    clarified = str(answer_text or "").strip()
    lowered = clarified.lower()
    kind = str((clarification or {}).get("kind") or "clarification").strip()
    level = str((clarification or {}).get("level") or "L2").strip()
    missing_slots_raw = (clarification or {}).get("missing_slots")
    missing_slots = [
        str(item).strip().lower()
        for item in (missing_slots_raw if isinstance(missing_slots_raw, list) else [])
        if str(item).strip()
    ]
    expected_answer = str((clarification or {}).get("expected_answer") or "").strip()
    selected_option = lowered if lowered and len(lowered) <= 80 else ""
    requested_scopes = _extract_scope_list_from_answer(clarified)

    context: dict[str, object] = {
        "round": max(0, int(round_no)),
        "kind": kind or "clarification",
        "level": level or "L2",
        "missing_slots": missing_slots,
        "expected_answer": expected_answer,
        "answer_text": clarified,
        "selected_option": selected_option,
        "requested_scopes": requested_scopes,
        "confirmed": lowered in {"si", "sí", "ok", "confirmo", "confirmado", "si, continuar"},
        "plan_approved": kind == "plan_approval"
        and lowered in {"si", "sí", "ok", "confirmo", "confirmado"},
    }

    if "scope" in missing_slots and not requested_scopes and clarified:
        context["objective_hint"] = clarified

    return context


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
