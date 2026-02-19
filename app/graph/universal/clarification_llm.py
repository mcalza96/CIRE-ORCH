from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = (
    "Eres un asistente de aclaracion para un orquestador RAG empresarial. "
    "Devuelve SOLO JSON valido con este esquema: "
    '{"question": "...", "options": ["..."], "missing_slots": ["scope"], '
    '"expected_answer": "...", "confidence": 0.0}. '
    "La pregunta debe ser breve, concreta, empatica y accionable. "
    "Si falta alcance, pide alcance exacto. "
    "No inventes hechos del dominio; solo ayuda a desambiguar."
)


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


async def build_clarification_with_llm(
    *,
    query: str,
    current_question: str,
    current_options: tuple[str, ...],
    missing_slots: list[str],
    scope_candidates: tuple[str, ...],
    interaction_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(getattr(settings, "ORCH_CLARIFICATION_LLM_ENABLED", True)):
        return None
    api_key = str(getattr(settings, "GROQ_API_KEY", "") or "").strip()
    if not api_key:
        return None

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    except Exception:
        return None

    model = str(getattr(settings, "ORCH_CLARIFICATION_MODEL", "") or "").strip()
    if not model:
        model = str(getattr(settings, "GROQ_MODEL_LIGHTWEIGHT", "") or "").strip()
    if not model:
        return None

    timeout_s = max(0.8, float(getattr(settings, "ORCH_CLARIFICATION_TIMEOUT_S", 2.0) or 2.0))
    max_options = max(2, int(getattr(settings, "ORCH_CLARIFICATION_MAX_OPTIONS", 4) or 4))
    prompt = {
        "query": str(query or "")[:500],
        "current_question": str(current_question or "")[:400],
        "current_options": list(current_options[:max_options]),
        "missing_slots": [str(slot) for slot in missing_slots[:4]],
        "scope_candidates": list(scope_candidates[:6]),
        "interaction_metrics": dict(interaction_metrics or {}),
        "instructions": (
            "Redacta una pregunta de aclaracion especifica para llenar slots faltantes. "
            "Si el slot faltante es scope y hay candidatos, ofrece opciones concretas. "
            "Si no hay candidatos, pide lista explicita separada por comas."
        ),
    }

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0.1,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.warning("clarification_llm_failed", error=str(exc))
        return None

    raw = str(completion.choices[0].message.content or "").strip()
    payload = _parse_json_payload(raw)
    if not payload:
        return None

    question = str(payload.get("question") or "").strip()
    options_raw = payload.get("options")
    options = (
        [str(item).strip() for item in options_raw if str(item).strip()]
        if isinstance(options_raw, list)
        else []
    )
    if not question:
        return None
    confidence = payload.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except Exception:
        conf = 0.0
    return {
        "question": question,
        "options": options[:max_options],
        "missing_slots": [str(item) for item in list(payload.get("missing_slots") or [])[:4]],
        "expected_answer": str(payload.get("expected_answer") or "").strip(),
        "confidence": conf,
        "model": model,
    }
