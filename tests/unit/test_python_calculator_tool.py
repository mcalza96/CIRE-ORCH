import pytest

from app.agent.tools.base import ToolRuntimeContext
from app.agent.tools.python_calculator import PythonCalculatorTool
from app.profiles.models import AgentProfile


class _DummyRetriever:
    async def retrieve_chunks(self, *args, **kwargs):
        del args, kwargs
        return []

    async def retrieve_summaries(self, *args, **kwargs):
        del args, kwargs
        return []

    async def validate_scope(self, *args, **kwargs):
        del args, kwargs
        return {}

    def apply_validated_scope(self, *args, **kwargs):
        del args, kwargs


class _DummyAnswer:
    async def generate(self, *args, **kwargs):
        del args, kwargs
        return None


class _DummyValidator:
    def validate(self, *args, **kwargs):
        del args, kwargs
        return None


def _ctx() -> ToolRuntimeContext:
    return ToolRuntimeContext(
        retriever=_DummyRetriever(),
        answer_generator=_DummyAnswer(),
        validator=_DummyValidator(),
    )


@pytest.mark.asyncio
async def test_python_calculator_computes_expression() -> None:
    tool = PythonCalculatorTool()
    result = await tool.run(
        {"expression": "5 * (samples + controls)", "variables": {"samples": 20, "controls": 2}},
        state={"agent_profile": AgentProfile(profile_id="lab")},
        context=_ctx(),
    )
    assert result.ok is True
    assert result.output["result"] == 110.0


@pytest.mark.asyncio
async def test_python_calculator_blocks_forbidden_nodes() -> None:
    tool = PythonCalculatorTool()
    result = await tool.run(
        {"expression": "__import__('os').system('whoami')"},
        state={"agent_profile": AgentProfile(profile_id="lab")},
        context=_ctx(),
    )
    assert result.ok is False
    assert "forbidden" in result.error
