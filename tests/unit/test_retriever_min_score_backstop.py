from types import SimpleNamespace
from unittest.mock import MagicMock

from app.agent.http_adapters import RagEngineRetrieverAdapter
from app.core.config import settings


def _build_adapter(min_score: float) -> RagEngineRetrieverAdapter:
    adapter = RagEngineRetrieverAdapter(contract_client=MagicMock())
    adapter._profile_context = SimpleNamespace(
        retrieval=SimpleNamespace(min_score=min_score)
    )
    return adapter


def test_min_score_backstop_keeps_top_candidates_when_all_below_threshold(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "ORCH_MIN_SCORE_BACKSTOP_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_MIN_SCORE_BACKSTOP_TOP_N", 2)

    adapter = _build_adapter(0.72)
    items = [
        {"source": "C1", "score": 0.69},
        {"source": "C2", "score": 0.66},
        {"source": "C3", "score": 0.55},
    ]
    trace: dict = {}
    kept = adapter._filter_items_by_min_score(items, trace_target=trace)

    assert [str(row.get("source")) for row in kept] == ["C1", "C2"]
    assert trace["min_score_filter"]["backstop_applied"] is True
    assert trace["min_score_filter"]["backstop_top_n"] == 2


def test_min_score_backstop_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ORCH_MIN_SCORE_BACKSTOP_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_MIN_SCORE_BACKSTOP_TOP_N", 2)

    adapter = _build_adapter(0.72)
    items = [{"source": "C1", "score": 0.69}, {"source": "C2", "score": 0.66}]
    trace: dict = {}
    kept = adapter._filter_items_by_min_score(items, trace_target=trace)

    assert kept == []
    assert trace["min_score_filter"]["backstop_applied"] is False
    assert trace["min_score_filter"]["backstop_top_n"] == 0
