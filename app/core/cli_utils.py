from __future__ import annotations

import re
import httpx

def rewrite_query_with_clarification(original_query: str, clarification_answer: str) -> str:
    text = (clarification_answer or "").strip()
    if not text:
        return original_query
    lowered = text.lower().strip()
    if re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", lowered):
        return f"{original_query}\n\n__clarified_mode__={lowered} Aclaracion de modo: {text}."
    coverage_tag = ""
    if lowered in {"respuesta parcial", "aceptar respuesta parcial"}:
        coverage_tag = "__coverage__=partial "
    elif lowered in {"cobertura completa", "exigir cobertura completa"}:
        coverage_tag = "__coverage__=full "
    return (
        f"{original_query}\n\n__clarified_scope__=true {coverage_tag}Aclaracion de alcance: {text}."
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
