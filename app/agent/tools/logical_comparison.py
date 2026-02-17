from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.components.parsing import extract_row_standard
from app.agent.models import ToolResult
from app.agent.tools.base import ToolRuntimeContext


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
        evidence = list(state.get("retrieved_documents") or [])
        grouped = _cluster_by_scope(evidence)
        if not grouped:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="missing_evidence_for_comparison",
            )

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

        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "topic": topic,
                "rows": rows,
                "comparison_markdown": "\n".join(md),
            },
        )
