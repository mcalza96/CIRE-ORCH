from __future__ import annotations

from dataclasses import dataclass

from app.agent.types.models import AnswerDraft, RetrievalPlan, ToolResult
from app.agent.tools.base import ToolRuntimeContext


@dataclass(frozen=True)
class CitationValidatorTool:
    name: str = "citation_validator"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        draft = state.get("generation")
        if not isinstance(draft, AnswerDraft):
            text = str(payload.get("draft_response") or "").strip()
            mode = str(payload.get("mode") or "default")
            draft = AnswerDraft(text=text, mode=mode)  # type: ignore[arg-type]

        plan = state.get("retrieval_plan")
        if not isinstance(plan, RetrievalPlan):
            plan = RetrievalPlan(
                mode=str(getattr(draft, "mode", "default")),  # type: ignore[arg-type]
                chunk_k=0,
                chunk_fetch_k=0,
                summary_k=0,
            )
        query = str(state.get("user_query") or payload.get("query") or "")
        validation = context.validator.validate(draft, plan, query)
        return ToolResult(
            tool=self.name,
            ok=bool(validation.accepted),
            output={
                "accepted": bool(validation.accepted),
                "issues": list(validation.issues),
            },
        )
