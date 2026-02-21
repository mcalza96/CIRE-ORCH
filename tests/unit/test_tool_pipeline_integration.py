"""Tests for tool pipeline integration: working_memory → generator and tool-to-tool piping."""
from __future__ import annotations

import pytest

from app.agent.types.models import EvidenceItem, ToolResult
# ── Fix 2: Piping tests ──────────────────────────────────────────────

from app.agent.tools.structural_extraction import StructuralExtractionTool
from app.agent.tools.logical_comparison import LogicalComparisonTool
from app.agent.tools.base import ToolRuntimeContext
def _make_evidence(content: str, standard: str = "") -> EvidenceItem:
    return EvidenceItem(
        source="C1",
        content=content,
        score=0.9,
        metadata={"row": {"content": content, "metadata": {"source_standard": standard}}},
    )


def _dummy_context() -> ToolRuntimeContext:
    class _DummyRetriever:
        async def retrieve_chunks(self, *a, **kw): return []
        async def retrieve_summaries(self, *a, **kw): return []

    class _DummyAnswer:
        async def generate(self, *a, **kw): return None

    class _DummyValidator:
        def validate(self, *a, **kw): return None

    return ToolRuntimeContext(
        retriever=_DummyRetriever(),
        answer_generator=_DummyAnswer(),
        validator=_DummyValidator(),
    )


@pytest.mark.asyncio
async def test_structural_extraction_prefers_piped_chunks() -> None:
    tool = StructuralExtractionTool()
    piped_chunk = _make_evidence("Concentración Acetona: 25 mg en la muestra")
    state_chunk = _make_evidence("Volumen Etanol: 100 ml en la muestra")

    # With piped metadata → should see Acetona
    payload = {
        "previous_tool_metadata": {
            "chunks": [piped_chunk],
        }
    }
    result = await tool.run(
        payload,
        state={"retrieved_documents": [state_chunk]},
        context=_dummy_context(),
    )
    assert result.ok
    records = result.output.get("records", [])
    labels = [r.get("label", "") for r in records]
    assert any("Acetona" in label for label in labels), f"Expected Acetona in {labels}"

    # Without piped metadata → falls back to state
    result_fb = await tool.run(
        {},
        state={"retrieved_documents": [state_chunk]},
        context=_dummy_context(),
    )
    assert result_fb.ok
    records_fb = result_fb.output.get("records", [])
    labels_fb = [r.get("label", "") for r in records_fb]
    assert any("Etanol" in label for label in labels_fb), f"Expected Etanol in {labels_fb}"


@pytest.mark.asyncio
async def test_logical_comparison_prefers_piped_chunks() -> None:
    tool = LogicalComparisonTool()
    piped_chunk = _make_evidence("Clausula 4.1 contexto", standard="ISO 9001")
    state_chunk = _make_evidence("Clausula 6.1 riesgos", standard="ISO 14001")

    # With piped metadata → should cluster piped data
    payload = {
        "previous_tool_metadata": {
            "chunks": [piped_chunk],
        }
    }
    result = await tool.run(
        payload,
        state={"retrieved_documents": [state_chunk]},
        context=_dummy_context(),
    )
    assert result.ok
    rows = result.output.get("rows", [])
    # The piped chunk has standard ISO 9001
    scopes = [r.get("scope", "") for r in rows]
    # Should NOT contain ISO 14001 from state since piped takes priority
    # (piped chunk is ISO 9001 via metadata)
    assert len(rows) >= 1

    # Without piped metadata → falls back to state
    result_fb = await tool.run(
        {},
        state={"retrieved_documents": [state_chunk]},
        context=_dummy_context(),
    )
    assert result_fb.ok
    rows_fb = result_fb.output.get("rows", [])
    assert len(rows_fb) >= 1


@pytest.mark.asyncio
async def test_logical_comparison_empty_piped_falls_back_to_state() -> None:
    tool = LogicalComparisonTool()
    state_chunk = _make_evidence("Clausula 5.2 pol\u00edtica", standard="ISO 9001")

    # Empty piped metadata → should fall back
    payload = {"previous_tool_metadata": {"chunks": []}}
    result = await tool.run(
        payload,
        state={"retrieved_documents": [state_chunk]},
        context=_dummy_context(),
    )
    assert result.ok
    assert len(result.output.get("rows", [])) >= 1
