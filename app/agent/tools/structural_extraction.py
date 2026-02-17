from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.models import ToolResult
from app.agent.tools.base import ToolRuntimeContext


_MEASURE_RE = re.compile(
    r"(?P<label>[A-Za-zÁÉÍÓÚáéíóúñÑ0-9 _/-]{2,60})\s*[:=-]?\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mg|g|kg|ml|mL|l|L|ug|µg|ppm|ppb|nm)\b"
)


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
        text_content = str(payload.get("text_content") or "").strip()
        if not text_content:
            fragments: list[str] = []
            for item in list(state.get("retrieved_documents") or []):
                content = str(getattr(item, "content", "") or "").strip()
                if content:
                    fragments.append(content)
                if len(fragments) >= 12:
                    break
            text_content = "\n".join(fragments).strip()

        schema_definition = str(payload.get("schema_definition") or payload.get("target_schema") or "").strip()
        if not text_content:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="empty_text_content",
            )

        records: list[dict[str, object]] = []
        for match in _MEASURE_RE.finditer(text_content):
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

        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "schema_definition": schema_definition,
                "records": records,
                "record_count": len(records),
            },
        )
