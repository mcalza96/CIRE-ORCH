from __future__ import annotations

from typing import Protocol

from app.agent.application import AnswerGeneratorPort, RetrieverPort, ValidatorPort
from app.agent.tools import AgentTool, ToolRuntimeContext


class OrchestratorComponents(Protocol):
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, AgentTool] | None

    def _runtime_context(self) -> ToolRuntimeContext: ...
