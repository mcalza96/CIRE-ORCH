from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.agent.engine import AnswerGeneratorPort, RetrieverPort, ValidatorPort
from app.agent.types.models import ToolResult


@dataclass(frozen=True)
class ToolRuntimeContext:
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort


class AgentTool(Protocol):
    name: str

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult: ...
