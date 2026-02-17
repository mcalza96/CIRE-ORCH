from __future__ import annotations

from dataclasses import dataclass

from app.agent.tools.base import AgentTool
from app.agent.tools.citation_validator import CitationValidatorTool
from app.agent.tools.logical_comparison import LogicalComparisonTool
from app.agent.tools.python_calculator import PythonCalculatorTool
from app.agent.tools.semantic_retrieval import SemanticRetrievalTool
from app.agent.tools.structural_extraction import StructuralExtractionTool
from app.cartridges.models import AgentProfile, ToolName


@dataclass(frozen=True)
class ToolRegistry:
    _tools: dict[str, AgentTool]

    @classmethod
    def create_default(cls) -> "ToolRegistry":
        tools: dict[str, AgentTool] = {
            "semantic_retrieval": SemanticRetrievalTool(),
            "structural_extraction": StructuralExtractionTool(),
            "logical_comparison": LogicalComparisonTool(),
            "python_calculator": PythonCalculatorTool(),
            "citation_validator": CitationValidatorTool(),
        }
        return cls(_tools=tools)

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def allowed_tools(self, profile: AgentProfile | None) -> list[str]:
        if profile is None:
            return ["semantic_retrieval", "citation_validator"]

        out: list[str] = []
        seen: set[str] = set()
        for raw in list(profile.capabilities.allowed_tools):
            name = str(raw).strip()
            if not name or name in seen:
                continue
            if name not in self._tools:
                continue
            policy = profile.capabilities.tool_policies.get(name)  # type: ignore[arg-type]
            if policy is not None and not bool(policy.enabled):
                continue
            seen.add(name)
            out.append(name)

        if not out:
            out = ["semantic_retrieval", "citation_validator"]
        return out

    def has(self, name: ToolName) -> bool:
        return str(name) in self._tools
