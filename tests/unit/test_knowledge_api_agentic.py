import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from app.api.v1.routers.knowledge import router as knowledge_router
from app.agent.engine import HandleQuestionResult
from app.agent.types.models import (
    AnswerDraft,
    EvidenceItem,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
)


@pytest.fixture
def mock_use_case():
    return AsyncMock()


@pytest.fixture
def client(mock_use_case):
    from app.api.v1.routers.knowledge import get_current_user, _build_use_case

    app = FastAPI()
    app.include_router(knowledge_router, prefix="/api/v1/knowledge")

    # Override EVERY dependency that might hit the wire
    app.dependency_overrides[get_current_user] = lambda: MagicMock(user_id="test-user")
    app.dependency_overrides[_build_use_case] = lambda: mock_use_case

    # Mock at the API route module level for authorize_requested_tenant
    with patch(
        "app.api.v1.routers.knowledge.authorize_requested_tenant",
        AsyncMock(return_value="test-tenant"),
    ):
        yield TestClient(app)


def test_answer_api_includes_retrieval_plan(client, mock_use_case):
    mock_result = HandleQuestionResult(
        intent=QueryIntent(mode="explicativa"),
        answer=AnswerDraft(text="Test answer", mode="explicativa", evidence=[]),
        plan=RetrievalPlan(
            mode="explicativa",
            chunk_k=10,
            chunk_fetch_k=50,
            summary_k=5,
            requested_standards=("ISO 9001",),
        ),
        retrieval=RetrievalDiagnostics(
            contract="advanced",
            strategy="multi_query_primary",
            trace={
                "promoted": True,
                "reason": "complex_intent",
                "subqueries": [{"id": "q1", "query": "..."}],
                "timings_ms": {"multi_query_primary": 120.5},
            },
        ),
        validation=MagicMock(accepted=True, issues=[]),
        clarification=None,
    )
    mock_use_case.execute = AsyncMock(return_value=mock_result)

    response = client.post(
        "/api/v1/knowledge/answer", json={"query": "test query", "tenant_id": "test-tenant"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "retrieval_plan" in data
    rp = data["retrieval_plan"]
    assert rp["promoted"] is True
    assert rp["reason"] == "complex_intent"
    assert len(rp["subqueries"]) == 1
    assert rp["timings_ms"]["multi_query_primary"] == 120.5
    assert data.get("engine") == "universal_flow"
    assert isinstance(data.get("reasoning_trace"), dict)


def test_answer_api_includes_standard_fields(client, mock_use_case):
    mock_result = HandleQuestionResult(
        intent=QueryIntent(mode="explicativa"),
        answer=AnswerDraft(text="Test answer", mode="explicativa", evidence=[]),
        plan=RetrievalPlan(mode="explicativa", chunk_k=10, chunk_fetch_k=50, summary_k=5),
        retrieval=RetrievalDiagnostics(contract="legacy"),
        validation=MagicMock(accepted=False, issues=["some issue"]),
        clarification=MagicMock(question="clarify?", options=("a", "b")),
    )
    mock_use_case.execute = AsyncMock(return_value=mock_result)

    response = client.post(
        "/api/v1/knowledge/answer", json={"query": "test query", "tenant_id": "test-tenant"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "validation" in data
    assert data["validation"]["accepted"] is False
    assert "clarification" in data
    assert data["clarification"]["question"] == "clarify?"
    assert "agent_profile" in data
    assert isinstance(data["agent_profile"].get("resolution"), dict)
    assert data.get("engine") == "universal_flow"
    assert isinstance(data.get("reasoning_trace"), dict)


def test_answer_stream_includes_context_chunks_count(client, mock_use_case):
    mock_result = HandleQuestionResult(
        intent=QueryIntent(mode="explicativa"),
        answer=AnswerDraft(
            text="Test answer",
            mode="explicativa",
            evidence=[EvidenceItem(source="C1", content="fragmento")],
        ),
        plan=RetrievalPlan(mode="explicativa", chunk_k=10, chunk_fetch_k=50, summary_k=5),
        retrieval=RetrievalDiagnostics(contract="advanced", strategy="langgraph_universal_flow"),
        validation=MagicMock(accepted=True, issues=[]),
        clarification=None,
    )
    mock_use_case.execute = AsyncMock(return_value=mock_result)

    response = client.post(
        "/api/v1/knowledge/answer/stream",
        json={"query": "test query", "tenant_id": "test-tenant"},
    )

    assert response.status_code == 200
    assert '"context_chunks_count": 1' in response.text
    assert '"context_chunks": ["fragmento"]' in response.text


def test_answer_api_passes_clarification_context_to_command(client, mock_use_case):
    captured = {}

    async def _execute(cmd):
        captured["context"] = cmd.clarification_context
        return HandleQuestionResult(
            intent=QueryIntent(mode="explicativa"),
            answer=AnswerDraft(text="ok", mode="explicativa", evidence=[]),
            plan=RetrievalPlan(mode="explicativa", chunk_k=10, chunk_fetch_k=50, summary_k=5),
            retrieval=RetrievalDiagnostics(contract="legacy"),
            validation=MagicMock(accepted=True, issues=[]),
            clarification=None,
        )

    mock_use_case.execute = AsyncMock(side_effect=_execute)

    response = client.post(
        "/api/v1/knowledge/answer",
        json={
            "query": "test query",
            "tenant_id": "test-tenant",
            "clarification_context": {
                "round": 1,
                "kind": "clarification",
                "requested_scopes": ["ISO 9001"],
                "answer_text": "ISO 9001",
            },
        },
    )

    assert response.status_code == 200
    assert isinstance(captured.get("context"), dict)
    assert captured["context"].get("requested_scopes") == ["ISO 9001"]
