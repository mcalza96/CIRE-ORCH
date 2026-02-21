from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from app.agent.components.parsing import extract_row_standard
from app.agent.models import ToolResult
from app.agent.tools.base import ToolRuntimeContext
from app.infrastructure.config import settings

logger = structlog.get_logger(__name__)


def _cluster_by_scope(rows: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in rows:
        scope = extract_row_standard(item)
        label = scope or "SIN_SCOPE"
        grouped.setdefault(label, [])
        content = str(getattr(item, "content", "") or "").strip()
        if content:
            grouped[label].append(content)
    return grouped


def _deterministic_comparison(
    topic: str,
    grouped: dict[str, list[str]],
) -> dict[str, object]:
    """Fast, deterministic groupby comparison (original logic)."""
    rows: list[dict[str, str]] = []
    for scope, contents in grouped.items():
        snippet = " ".join(" ".join(text.split()) for text in contents[:2]).strip()
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "..."
        rows.append({"scope": scope, "evidence": snippet or "sin_evidencia"})
    rows.sort(key=lambda item: item["scope"])
    md = ["| Scope | Evidence |", "|---|---|"]
    for row in rows:
        md.append(f"| {row['scope']} | {row['evidence']} |")
    return {
        "topic": topic,
        "rows": rows,
        "comparison_markdown": "\n".join(md),
    }


_COMPARISON_SYSTEM = (
    "Eres un analista técnico que compara evidencia entre distintas normas o ámbitos. "
    "Devuelve SOLO JSON con el schema: "
    '{"analysis": "texto con análisis comparativo", "commonalities": ["..."], '
    '"differences": ["..."], "gaps": ["ámbito o norma sin evidencia suficiente"]}'
)

_COMPARISON_MAX_CONTEXT_CHARS = 3000


async def _llm_comparison(
    topic: str,
    grouped: dict[str, list[str]],
) -> dict[str, object] | None:
    """LLM-powered comparison synthesis. Returns None on failure (graceful degradation)."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return None

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    except Exception:
        return None

    # Build compact context per scope
    context_parts: list[str] = []
    total_chars = 0
    for scope, contents in grouped.items():
        snippet = " ".join(" ".join(t.split()) for t in contents[:3]).strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        entry = f"[{scope}]\n{snippet}"
        if total_chars + len(entry) > _COMPARISON_MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    user_msg = (
        f"TEMA: {topic}\n\n"
        f"EVIDENCIA POR ÁMBITO:\n{'---'.join(context_parts)}\n\n"
        "Analiza comparativamente la evidencia entre los ámbitos."
    )

    timeout_s = max(
        2.0,
        float(getattr(settings, "ORCH_TOOL_LLM_TIMEOUT_S", 4.0) or 4.0),
    )

    try:
        import asyncio

        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.GROQ_MODEL_LIGHTWEIGHT,
                temperature=0.0,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": _COMPARISON_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            ),
            timeout=timeout_s,
        )
        text = (completion.choices[0].message.content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning("logical_comparison_llm_failed", error=str(exc))

    return None


@dataclass(frozen=True)
class LogicalComparisonTool:
    name: str = "logical_comparison"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        del context
        topic = str(payload.get("topic") or state.get("working_query") or "").strip()
        # Priority 1: chunks piped from the previous tool (e.g. semantic_retrieval)
        evidence: list[Any] = []
        prev_meta = payload.get("previous_tool_metadata")
        if isinstance(prev_meta, dict):
            prev_chunks = list(prev_meta.get("chunks") or [])
            if prev_chunks:
                evidence = prev_chunks
        # Priority 2: fallback to state global
        if not evidence:
            evidence = list(state.get("retrieved_documents") or [])
        grouped = _cluster_by_scope(evidence)
        if not grouped:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="missing_evidence_for_comparison",
            )

        # Deterministic base (always computed)
        deterministic = _deterministic_comparison(topic, grouped)

        # LLM enhancement (best-effort)
        llm_result = await _llm_comparison(topic, grouped)
        if isinstance(llm_result, dict):
            deterministic["llm_analysis"] = str(llm_result.get("analysis") or "")
            deterministic["commonalities"] = list(llm_result.get("commonalities") or [])
            deterministic["differences"] = list(llm_result.get("differences") or [])
            deterministic["gaps"] = list(llm_result.get("gaps") or [])
            deterministic["llm_enhanced"] = True
        else:
            deterministic["llm_enhanced"] = False

        return ToolResult(
            tool=self.name,
            ok=True,
            output=deterministic,
        )
