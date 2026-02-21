from __future__ import annotations

from typing import Mapping

from app.agent.tools.base import AgentTool
from app.agent.tools.citation_validator import CitationValidatorTool
from app.agent.tools.expectation_coverage import ExpectationCoverageTool
from app.agent.tools.logical_comparison import LogicalComparisonTool
from app.agent.tools.python_calculator import PythonCalculatorTool
from app.agent.tools.semantic_retrieval import SemanticRetrievalTool
from app.agent.tools.structural_extraction import StructuralExtractionTool
from app.profiles.models import AgentProfile


DEFAULT_ALLOWED_TOOLS = ("semantic_retrieval", "citation_validator")


def create_default_tools() -> dict[str, AgentTool]:
    return {
        "semantic_retrieval": SemanticRetrievalTool(),
        "structural_extraction": StructuralExtractionTool(),
        "logical_comparison": LogicalComparisonTool(),
        "expectation_coverage": ExpectationCoverageTool(),
        "python_calculator": PythonCalculatorTool(),
        "citation_validator": CitationValidatorTool(),
    }


def get_tool(tools: Mapping[str, AgentTool], name: str) -> AgentTool | None:
    return tools.get(name)


def resolve_allowed_tools(
    profile: AgentProfile | None, tools: Mapping[str, AgentTool]
) -> list[str]:
    if profile is None:
        return [name for name in DEFAULT_ALLOWED_TOOLS if name in tools]

    out: list[str] = []
    seen: set[str] = set()
    for configured_name in list(profile.capabilities.allowed_tools):
        name = str(configured_name).strip()
        if not name or name in seen:
            continue
        if name not in tools:
            continue
        policy = profile.capabilities.tool_policies.get(configured_name)
        if policy is not None and not bool(policy.enabled):
            continue
        seen.add(name)
        out.append(name)

    if not out:
        out = [name for name in DEFAULT_ALLOWED_TOOLS if name in tools]
    return out
