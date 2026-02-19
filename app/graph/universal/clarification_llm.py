from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = (
    "Eres un asistente de aclaracion para un orquestador RAG empresarial. "
    "DEBES devolver UNICAMENTE un objeto JSON válido con este exacto esquema:\n"
    '{"question": "...", "options": ["..."], "missing_slots": ["scope"], '
    '"expected_answer": "...", "confidence": 0.0}\n'
    "La pregunta debe ser breve, concreta, empatica y accionable. "
    "Si falta alcance, pide alcance exacto. "
    "No uses ejemplos de normas ISO salvo que el query mencione ISO de forma explicita. "
    "No inventes hechos del dominio; solo ayuda a desambiguar. "
    "No devuelvas ningún texto antes ni después del JSON. No uses bloques markdown."
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
        logger.info("clarification_llm_skipped", reason="missing_groq_api_key")
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
            "Redacta una pregunta de aclaracion breve, natural y conversacional para solucionar la ambiguedad llenando EXACTAMENTE los 'missing_slots'. "
            "Si falta 'scope' y hay candidatos, ofrece las opciones concretas. "
            "Si falta 'objective', preguntale en que area o tema en especifico desea centrar el analisis. "
            "Nunca incluyas ejemplos que se salgan del giro o dominio mencionado en el query."
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
                response_format={"type": "json_object"},
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
    confidence = payload.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(str(confidence if confidence is not None else 0.0))))
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

async def extract_clarification_slots_with_llm(
    *,
    clarification_text: str,
    original_query: str,
    missing_slots: list[str],
) -> dict[str, list[str]] | None:
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
    prompt = {
        "original_query": str(original_query or "")[:500],
        "missing_slots": [str(s) for s in missing_slots[:4]],
        "user_answer": str(clarification_text or "")[:400],
        "instructions": (
            "Extrae los valores para cada uno de los slots indicados en 'missing_slots'."
            "Devuelve la informacion como un objeto JSON donde cada clave es el nombre del slot"
            "y el valor es una lista homogenea con los valores extraidos. Si es 'scope', "
            "normaliza al estandar (ej: 'ISO 45001'). Si no hay datos claros para un slot, "
            "retorna una lista vacia para esa clave."
        ),
    }

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": "Eres un extractor de entidades. DEBES devolver UNICAMENTE un objeto JSON válido, ej: {'scope': ['ISO 9001'], 'target_clauses': ['5.1', '5.2']}. No devuelvas texto adicional ni uses markdown."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.warning("clarification_extractor_llm_failed", error=str(exc))
        return None
        
    raw = str(completion.choices[0].message.content or "").strip()
    payload = _parse_json_payload(raw)
    if not payload:
        return None
        
    return payload

async def rewrite_plan_with_feedback_llm(
    *,
    current_tools: list[str],
    feedback: str,
    allowed_tools: list[str],
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
    
    timeout_s = max(1.0, float(getattr(settings, "ORCH_CLARIFICATION_TIMEOUT_S", 2.0) or 2.0))
    prompt = {
        "current_plan": current_tools,
        "user_feedback": str(feedback or "")[:500],
        "allowed_tools": allowed_tools,
        "instructions": (
            "El usuario ha rechazado el plan propuesto o pidio ajustes. "
            "Reescribe el plan devolviendo una lista de nombres de herramientas ('new_plan') usando SOLO las de 'allowed_tools'. "
            "Si pide estructurar/tabla/extraer, añade 'structural_extraction'. "
            "Si pide cancelar comparaciones, elimina 'logical_comparison'. "
            "Ademas, provee 'dynamic_inputs' si aplica (ej. para structural_extraction define un 'schema_definition' segun lo que pidio el usuario). "
            "Respeta la secuencia logica: primero traer datos, luego procesarlos."
        )
    }

    schema_example = '{"new_plan": ["semantic_retrieval", "structural_extraction"], "dynamic_inputs": {"structural_extraction": {"schema_definition": "roles, responsabilidades"}}}'

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=800,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres el planificador L3. DEBES devolver UNICAMENTE un objeto JSON válido con este exacto esquema:\n"
                                   '{"new_plan": ["...", "..."], "dynamic_inputs": {"herramienta": {"parametro": "valor"}}}\n'
                                   "No devuelvas ningún texto antes ni después del JSON ni uses bloques markdown."
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.warning("replan_llm_failed", error=str(exc))
        return None
        
    raw = str(completion.choices[0].message.content or "").strip()
    payload = _parse_json_payload(raw)
    return payload
