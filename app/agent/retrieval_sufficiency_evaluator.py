from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.core.config import settings


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    reason: str = ""


@dataclass
class RetrievalSufficiencyEvaluator:
    """Optional LLM evaluator; kernel remains in control."""

    timeout_seconds: float = 6.0

    def __post_init__(self) -> None:
        self._client = (
            AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            if settings.GROQ_API_KEY
            else None
        )

    async def evaluate(
        self,
        *,
        query: str,
        requested_standards: tuple[str, ...],
        items: list[dict[str, Any]],
        min_items: int,
    ) -> SufficiencyDecision:
        if len(items) >= max(1, int(min_items)):
            return SufficiencyDecision(sufficient=True, reason="min_items_met")

        if self._client is None:
            return SufficiencyDecision(sufficient=False, reason="no_provider")

        model = (
            settings.ORCH_EVALUATOR_MODEL
            or settings.ORCH_PLANNER_MODEL
            or settings.GROQ_MODEL_LIGHTWEIGHT
        )

        # Keep input small: take only light metadata.
        sample = []
        for it in items[:10]:
            if not isinstance(it, dict):
                continue
            meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
            row = meta.get("row") if isinstance(meta.get("row"), dict) else {}
            sample.append(
                {
                    "id": it.get("source"),
                    "score": it.get("score"),
                    "source_standard": (row.get("metadata") or {}).get("source_standard")
                    if isinstance(row.get("metadata"), dict)
                    else None,
                    "preview": str(it.get("content") or "")[:160],
                }
            )

        system = (
            "Eres un evaluador de suficiencia para un sistema RAG. "
            "No respondas la pregunta. Solo decide si la evidencia recuperada parece suficiente."
        )
        user = (
            "Devuelve SOLO JSON con este schema:\n"
            '{"sufficient": true|false, "reason": "..."}\n\n'
            f"PREGUNTA: {query}\n"
            f"NORMAS OBJETIVO: {', '.join(requested_standards) if requested_standards else '(no especificadas)'}\n"
            f"EVIDENCIA (muestra):\n{json.dumps(sample, ensure_ascii=True)}\n\n"
            "Criterio: suficiente si hay cobertura razonable de las normas/clausulas pedidas y chunks relevantes."
        )

        try:
            completion = await self._client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < 0:
                return SufficiencyDecision(sufficient=False, reason="evaluator_invalid_json")
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict) and isinstance(data.get("sufficient"), bool):
                return SufficiencyDecision(
                    sufficient=bool(data["sufficient"]), reason=str(data.get("reason") or "")
                )
        except Exception as exc:
            logger.warning("sufficiency_evaluator_failed", error=str(exc))

        return SufficiencyDecision(sufficient=False, reason="evaluator_failed")
