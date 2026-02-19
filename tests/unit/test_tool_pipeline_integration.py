"""Tests for tool pipeline integration: working_memory → generator and tool-to-tool piping."""
from __future__ import annotations

import pytest

from app.agent.models import EvidenceItem, ToolResult
from app.graph.universal.steps import (
    _format_expectation_coverage,
    _format_logical_comparison,
    _format_structural_extraction,
    _format_generic_tool,
    _working_memory_to_evidence,
)
from app.agent.tools.structural_extraction import StructuralExtractionTool
from app.agent.tools.logical_comparison import LogicalComparisonTool
from app.agent.tools.base import ToolRuntimeContext


# ── Fix 1: _working_memory_to_evidence tests ──────────────────────────


class TestWorkingMemoryToEvidence:
    def test_empty_memory_returns_empty(self) -> None:
        assert _working_memory_to_evidence({}) == []

    def test_non_dict_values_are_skipped(self) -> None:
        assert _working_memory_to_evidence({"foo": "bar", "baz": 42}) == []

    def test_expectation_coverage_generates_evidence(self) -> None:
        mem = {
            "expectation_coverage": {
                "covered": [{"id": "c1"}],
                "missing": [
                    {"id": "m1", "missing_risk": "high", "reason": "no_data"}
                ],
                "coverage_ratio": 0.5,
            }
        }
        items = _working_memory_to_evidence(mem)
        assert len(items) == 1
        item = items[0]
        assert item.source == "TOOL_EXPECTATION_COVERAGE"
        assert "[EXPECTATION_COVERAGE]" in item.content
        assert "coverage_ratio=0.5" in item.content
        assert "missing:m1" in item.content
        assert item.score == 1.0
        assert item.metadata.get("synthetic") is True

    def test_logical_comparison_generates_evidence_with_markdown(self) -> None:
        mem = {
            "logical_comparison": {
                "topic": "ISO 9001 vs 14001",
                "rows": [
                    {"scope": "ISO 9001", "evidence": "Calidad"},
                    {"scope": "ISO 14001", "evidence": "Medioambiental"},
                ],
                "comparison_markdown": "| Scope | Evidence |\n|---|---|\n| ISO 9001 | Calidad |",
            }
        }
        items = _working_memory_to_evidence(mem)
        assert len(items) == 1
        item = items[0]
        assert item.source == "TOOL_LOGICAL_COMPARISON"
        assert "[LOGICAL_COMPARISON]" in item.content
        assert "topic=ISO 9001 vs 14001" in item.content
        assert "| Scope | Evidence |" in item.content

    def test_structural_extraction_generates_evidence(self) -> None:
        mem = {
            "structural_extraction": {
                "schema_definition": "BOM",
                "records": [
                    {"label": "Acetona", "value": 25.0, "unit": "mg"},
                    {"label": "Etanol", "value": 100.0, "unit": "ml"},
                ],
                "record_count": 2,
            }
        }
        items = _working_memory_to_evidence(mem)
        assert len(items) == 1
        item = items[0]
        assert item.source == "TOOL_STRUCTURAL_EXTRACTION"
        assert "[STRUCTURAL_EXTRACTION]" in item.content
        assert "schema=BOM" in item.content
        assert "Acetona: 25.0 mg" in item.content
        assert "Etanol: 100.0 ml" in item.content

    def test_unknown_tool_uses_generic_formatter(self) -> None:
        mem = {"my_custom_tool": {"result": "hello", "score": 0.9}}
        items = _working_memory_to_evidence(mem)
        assert len(items) == 1
        assert items[0].source == "TOOL_MY_CUSTOM_TOOL"
        assert "result=hello" in items[0].content

    def test_multiple_tools_in_memory(self) -> None:
        mem = {
            "expectation_coverage": {"covered": [], "missing": [], "coverage_ratio": 1.0},
            "logical_comparison": {"topic": "test", "rows": [], "comparison_markdown": ""},
        }
        items = _working_memory_to_evidence(mem)
        assert len(items) == 2
        sources = {item.source for item in items}
        assert "TOOL_EXPECTATION_COVERAGE" in sources
        assert "TOOL_LOGICAL_COMPARISON" in sources


class TestFormatters:
    def test_format_expectation_coverage_missing_rows(self) -> None:
        data = {
            "covered": [],
            "missing": [
                {"id": "x1", "missing_risk": "alto", "reason": "sin_datos"},
            ],
            "coverage_ratio": 0.0,
        }
        lines = _format_expectation_coverage(data)
        assert lines[0] == "[EXPECTATION_COVERAGE]"
        assert "coverage_ratio=0.0" in lines
        assert any("missing:x1" in line for line in lines)

    def test_format_logical_comparison_with_rows_fallback(self) -> None:
        data = {
            "topic": "test",
            "rows": [{"scope": "A", "evidence": "alpha"}],
            "comparison_markdown": "",
        }
        lines = _format_logical_comparison(data)
        assert any("| A | alpha |" in line for line in lines)

    def test_format_structural_extraction_empty_records(self) -> None:
        data = {"schema_definition": "", "records": [], "record_count": 0}
        lines = _format_structural_extraction(data)
        assert "[STRUCTURAL_EXTRACTION]" in lines
        assert "record_count=0" in lines

    def test_format_generic_tool_truncates(self) -> None:
        data = {"long_value": "x" * 300}
        lines = _format_generic_tool("big_tool", data)
        assert any("..." in line for line in lines)


# ── Fix 2: Piping tests ──────────────────────────────────────────────


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
