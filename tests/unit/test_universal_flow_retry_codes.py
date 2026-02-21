from app.agent.error_codes import RETRIEVAL_CODE_CLAUSE_MISSING
from app.agent.models import RetrievalDiagnostics, ToolResult
from app.infrastructure.config import settings
from app.graph.universal.logic import _extract_retry_signal_from_retrieval
from app.graph.universal.utils import _effective_execute_tool_timeout_ms


def test_extract_retry_signal_prefers_structured_error_codes() -> None:
    state = {
        "retrieval": RetrievalDiagnostics(
            contract="advanced",
            strategy="multi_query",
            trace={"error_codes": [RETRIEVAL_CODE_CLAUSE_MISSING]},
        )
    }
    last = ToolResult(
        tool="semantic_retrieval",
        ok=True,
        output={"chunk_count": 2, "summary_count": 1},
    )
    assert _extract_retry_signal_from_retrieval(state, last) == RETRIEVAL_CODE_CLAUSE_MISSING


def test_effective_execute_timeout_extends_semantic_retrieval_in_advanced_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "ORCH_RETRIEVAL_CONTRACT", "advanced", raising=False)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_EXECUTE_TOOL_MS", 2500)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_RETRIEVAL_HYBRID_MS", 10000)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_RETRIEVAL_MULTI_QUERY_MS", 9000)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COVERAGE_REPAIR_MS", 800)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_TOTAL_MS", 12000)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_PLAN_MS", 150)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_CLASSIFY_MS", 100)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_GENERATE_MS", 2000)
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_VALIDATE_MS", 700)

    assert _effective_execute_tool_timeout_ms("semantic_retrieval") == 8750


def test_effective_execute_timeout_keeps_base_for_non_semantic_or_legacy(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ORCH_TIMEOUT_EXECUTE_TOOL_MS", 2500)
    monkeypatch.setattr(settings, "ORCH_RETRIEVAL_CONTRACT", "legacy", raising=False)
    assert _effective_execute_tool_timeout_ms("semantic_retrieval") == 2500
    assert _effective_execute_tool_timeout_ms("python_calculator") == 2500
