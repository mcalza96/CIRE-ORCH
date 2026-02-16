from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.rag_contract_schemas import SubQueryRequest


logger = structlog.get_logger(__name__)


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start : end + 1]


@dataclass
class SemanticSubqueryPlanner:
    """LLM-based planner that proposes subqueries; kernel validates and enforces limits."""

    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self._client = (
            AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            if settings.GROQ_API_KEY
            else None
        )

    async def plan(
        self,
        *,
        query: str,
        requested_standards: tuple[str, ...] = (),
        max_queries: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of subquery dicts compatible with /retrieval/multi-query."""

        limit = max(1, min(int(max_queries or settings.ORCH_PLANNER_MAX_QUERIES or 5), 8))
        if not query.strip():
            return []

        # No provider configured: keep system usable.
        if self._client is None:
            logger.warning("semantic_planner_disabled_missing_key")
            return []

        model = settings.ORCH_PLANNER_MODEL or settings.GROQ_MODEL_ORCHESTRATION

        standards_text = (
            ", ".join(requested_standards) if requested_standards else "(no especificadas)"
        )
        system = (
            "Eres un planificador de busqueda para un sistema RAG. "
            "Tu trabajo NO es responder la pregunta: solo proponer sub-consultas de recuperacion. "
            "Devuelve JSON estricto. No incluyas texto extra."
        )
        user = (
            "Genera un plan de recuperacion en JSON con este schema exacto:\n"
            "{\n"
            '  "subqueries": [\n'
            '    {"id": "...", "query": "...", "k": null, "fetch_k": null, "filters": {\n'
            '       "source_standard": null, "source_standards": null, "metadata": null, "time_range": null\n'
            "    }}\n"
            "  ]\n"
            "}\n\n"
            f"REGLAS:\n- Maximo {limit} subqueries.\n- id corto (snake_case).\n"
            "- filters solo puede contener: source_standard, source_standards, metadata, time_range.\n"
            '- time_range, si lo usas, debe ser: {"field": "updated_at"|"created_at", "from": <iso8601>|null, "to": <iso8601>|null }.\n'
            "- Si hay alcances explicitos, crea al menos una subquery por alcance.\n"
            "- No inventes IDs de documentos ni tenants.\n\n"
            f"PREGUNTA: {query}\n"
            f"NORMAS DETECTADAS: {standards_text}\n"
        )

        async def call(messages: list[dict[str, str]]) -> str:
            completion = await self._client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=messages,
            )
            return (completion.choices[0].message.content or "").strip()

        raw = await call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        extracted = _extract_json_object(raw)
        if not extracted:
            logger.warning("semantic_planner_no_json", preview=raw[:120])
            return []

        try:
            payload = json.loads(extracted)
        except Exception:
            # Repair once.
            repair = await call(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "Repara este output para que sea JSON valido con el schema requerido. Devuelve SOLO JSON.\n\n"
                        + raw,
                    },
                ]
            )
            extracted2 = _extract_json_object(repair)
            if not extracted2:
                logger.warning("semantic_planner_repair_no_json")
                return []
            try:
                payload = json.loads(extracted2)
            except Exception:
                logger.warning("semantic_planner_repair_parse_failed")
                return []

        subq = payload.get("subqueries") if isinstance(payload, dict) else None
        if not isinstance(subq, list):
            return []

        # Kernel validation: drop invalid, enforce limit.
        out: list[dict[str, Any]] = []
        for raw_item in subq:
            if len(out) >= limit:
                break
            if not isinstance(raw_item, dict):
                continue
            try:
                item = SubQueryRequest.model_validate(raw_item)
            except Exception:
                continue
            out.append(item.model_dump(by_alias=True, exclude_none=True))

        return out
