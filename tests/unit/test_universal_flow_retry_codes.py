from app.agent.error_codes import RETRIEVAL_CODE_CLAUSE_MISSING
from app.agent.models import RetrievalDiagnostics, ToolResult
from app.graph.universal_flow import _extract_retry_signal_from_retrieval


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
