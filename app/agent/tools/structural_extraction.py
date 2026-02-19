from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog

from app.agent.models import ToolResult
from app.agent.tools.base import ToolRuntimeContext
from app.core.config import settings

logger = structlog.get_logger(__name__)


_MEASURE_RE = re.compile(
    r"(?P<label>[A-Za-zÁÉÍÓÚáéíóúñÑ0-9 _/-]{2,60})\s*[:=-]?\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mg|g|kg|ml|mL|l|L|ug|µg|ppm|ppb|nm)\b"
)


def _regex_extract(text: str) -> list[dict[str, object]]:
    """Deterministic regex extraction (original logic)."""
    records: list[dict[str, object]] = []
    for match in _MEASURE_RE.finditer(text):
        raw_value = str(match.group("value")).replace(",", ".")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        records.append(
            {
                "label": " ".join(match.group("label").split()),
                "value": value,
                "unit": match.group("unit"),
            }
        )
        if len(records) >= 100:
            break
    return records


_EXTRACTION_SYSTEM = (
    "Eres un extractor de datos estructurados de documentos técnicos. "
    "Devuelve SOLO JSON con el schema: "
    '{"records": [{"label": "nombre del dato", "value": 123.4, "unit": "unidad"}], '
    '"tables": [{"title": "título", "headers": ["col1", "col2"], '
    '"rows": [["val1", "val2"]]}], '
    '"key_values": [{"key": "nombre", "value": "valor"}]}'
)

_EXTRACTION_MAX_CONTEXT_CHARS = 3000


async def _llm_extract(
    text: str,
    schema_definition: str,
) -> dict[str, Any] | None:
    """LLM-powered structured extraction. Returns None on failure."""
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

    truncated = text[:_EXTRACTION_MAX_CONTEXT_CHARS]
    user_msg = (
        f"DOCUMENTO:\n{truncated}\n\n"
    )
    if schema_definition:
        user_msg += f"ESQUEMA OBJETIVO: {schema_definition}\n\n"
    user_msg += (
        "Extrae todos los datos estructurados: mediciones con unidades, "
        "tablas, y pares clave-valor relevantes."
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
                max_tokens=800,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            ),
            timeout=timeout_s,
        )
        raw = (completion.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning("structural_extraction_llm_failed", error=str(exc))

    return None


def _resolve_text_content(payload: dict[str, object], state: dict[str, object]) -> str:
    """Resolve text_content from payload (direct / piped) or state fallback."""
    text_content = str(payload.get("text_content") or "").strip()
    if not text_content:
        # Priority 1: chunks piped from the previous tool (e.g. semantic_retrieval)
        prev_meta = payload.get("previous_tool_metadata")
        if isinstance(prev_meta, dict):
            prev_chunks = list(prev_meta.get("chunks") or [])
            if prev_chunks:
                fragments: list[str] = []
                for item in prev_chunks[:12]:
                    content = str(getattr(item, "content", "") or "").strip()
                    if content:
                        fragments.append(content)
                text_content = "\n".join(fragments).strip()
        # Priority 2: fallback to state global
        if not text_content:
            fragments_fb: list[str] = []
            for item in list(state.get("retrieved_documents") or []):
                content = str(getattr(item, "content", "") or "").strip()
                if content:
                    fragments_fb.append(content)
                if len(fragments_fb) >= 12:
                    break
            text_content = "\n".join(fragments_fb).strip()
    return text_content


@dataclass(frozen=True)
class StructuralExtractionTool:
    name: str = "structural_extraction"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        del context
        text_content = _resolve_text_content(payload, state)

        schema_definition = str(
            payload.get("schema_definition") or payload.get("target_schema") or ""
        ).strip()
        if not text_content:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="empty_text_content",
            )

        # Layer 1: deterministic regex extraction (always runs)
        records = _regex_extract(text_content)

        # Layer 2: LLM-powered extraction (best-effort enhancement)
        llm_data = await _llm_extract(text_content, schema_definition)
        tables: list[dict[str, object]] = []
        key_values: list[dict[str, str]] = []
        llm_enhanced = False

        if isinstance(llm_data, dict):
            llm_enhanced = True
            # Merge LLM records with regex records (LLM may find records regex missed)
            llm_records = llm_data.get("records")
            if isinstance(llm_records, list):
                existing_labels = {str(r.get("label", "")).lower() for r in records}
                for rec in llm_records:
                    if not isinstance(rec, dict):
                        continue
                    label = str(rec.get("label") or "").strip()
                    if label.lower() not in existing_labels:
                        records.append(
                            {
                                "label": label,
                                "value": rec.get("value"),
                                "unit": str(rec.get("unit") or ""),
                                "source": "llm",
                            }
                        )
            # Tables and key-value pairs (LLM-only)
            raw_tables = llm_data.get("tables")
            if isinstance(raw_tables, list):
                tables = [t for t in raw_tables if isinstance(t, dict)][:10]
            raw_kvs = llm_data.get("key_values")
            if isinstance(raw_kvs, list):
                key_values = [kv for kv in raw_kvs if isinstance(kv, dict)][:20]

        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "schema_definition": schema_definition,
                "records": records,
                "record_count": len(records),
                "tables": tables,
                "key_values": key_values,
                "llm_enhanced": llm_enhanced,
            },
        )
