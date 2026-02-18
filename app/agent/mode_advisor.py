from __future__ import annotations

import json
from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI

from app.cartridges.models import AgentProfile
from app.agent.models import QueryMode
from app.core.config import settings


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AdvisorSuggestion:
    mode: QueryMode
    confidence: float
    rationale: str = ""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class ModeAdvisor:
    """Optional LLM advisor for mode selection. Kernel enforces guardrails."""

    def __post_init__(self) -> None:
        self._client = (
            AsyncOpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
            if settings.GROQ_API_KEY
            else None
        )

    async def suggest(
        self,
        *,
        query: str,
        candidate_modes: tuple[QueryMode, ...],
        profile: AgentProfile | None = None,
    ) -> AdvisorSuggestion | None:
        if self._client is None:
            return None
        model = settings.ORCH_MODE_ADVISOR_MODEL or settings.GROQ_MODEL_LIGHTWEIGHT
        system = (
            "Eres un clasificador de intencion para un sistema RAG. "
            "Elige el modo de respuesta mas adecuado para la pregunta. "
            "Devuelve SOLO JSON valido."
        )
        reference_patterns = list(profile.router.reference_patterns) if profile is not None else []
        scope_labels = list(profile.router.scope_hints.keys()) if profile is not None else []
        mode_hints: list[str] = []
        if profile is not None and profile.query_modes.modes:
            for mode_name in candidate_modes:
                cfg = profile.query_modes.modes.get(str(mode_name))
                if cfg is None:
                    continue
                mode_hints.append(
                    "- "
                    + str(mode_name)
                    + f": literal_evidence={bool(cfg.require_literal_evidence)}, allow_inference={bool(cfg.allow_inference)}, tool_hints={list(cfg.tool_hints)}"
                )

        user = (
            "Elige 1 modo de esta lista exacta: " + ", ".join(candidate_modes) + "\n\n"
            "Criterios por modo (desde cartucho):\n"
            + (
                "\n".join(mode_hints)
                if mode_hints
                else "- Usa el modo mas apropiado para la intencion detectada."
            )
            + "\n\n"
            f"Patrones de referencia activos: {reference_patterns or ['(default)']}\n"
            f"Alcances sugeridos: {scope_labels or ['(sin hints)']}\n\n"
            "Devuelve JSON schema:\n"
            '{"mode": <string>, "confidence": <0..1>, "rationale": <string>}\n\n'
            f"PREGUNTA: {query}"
        )
        try:
            completion = await self._client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            text = (completion.choices[0].message.content or "").strip()
            obj = _extract_json(text)
            if not obj:
                return None
            mode = str(obj.get("mode") or "").strip()
            if mode not in set(candidate_modes):
                return None
            conf = float(obj.get("confidence") or 0.0)
            conf = max(0.0, min(1.0, conf))
            rationale = str(obj.get("rationale") or "").strip()
            return AdvisorSuggestion(mode=mode, confidence=conf, rationale=rationale)
        except Exception as exc:
            logger.warning("mode_advisor_failed", error=str(exc))
            return None
