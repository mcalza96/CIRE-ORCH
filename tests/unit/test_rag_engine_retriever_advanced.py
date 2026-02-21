import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.infrastructure.clients.http_adapters import RagEngineRetrieverAdapter
from app.agent.types.models import RetrievalPlan, EvidenceItem


@pytest.fixture
def mock_contract_client():
    return MagicMock()


@pytest.fixture
def retriever(mock_contract_client):
    adapter = RagEngineRetrieverAdapter(contract_client=mock_contract_client)
    adapter._validated_filters = None
    return adapter


def create_retrieval_plan(mode="comparativa", standards=()):
    return RetrievalPlan(
        mode=mode,
        chunk_k=12,
        chunk_fetch_k=60,
        summary_k=5,
        require_literal_evidence=False,
        requested_standards=standards,
    )


@pytest.mark.asyncio
async def test_retrieve_advanced_multi_query_primary_success(retriever, mock_contract_client):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = True
        mock_settings.ORCH_MULTI_QUERY_MIN_ITEMS = 3
        mock_settings.ORCH_SEMANTIC_PLANNER = False  # Simplify for this test
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))

        mock_contract_client.multi_query = AsyncMock(
            return_value={
                "items": [
                    {"content": "c1", "source": "S1"},
                    {"content": "c2", "source": "S2"},
                    {"content": "c3", "source": "S3"},
                ],
                "trace": {"mq": "trace"},
                "subqueries": [],
            }
        )

        results = await retriever.retrieve_chunks(
            query="test multihop query about 9001 and 14001",
            tenant_id="t1",
            collection_id="c1",
            plan=plan,
        )

        assert len(results) == 3
        assert retriever.last_retrieval_diagnostics.strategy == "multi_query_primary"
        assert "multi_query_primary" in retriever.last_retrieval_diagnostics.trace["timings_ms"]


@pytest.mark.asyncio
async def test_retrieve_advanced_passes_semantic_tail_flag_to_planner(
    retriever, mock_contract_client
):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = True
        mock_settings.ORCH_MULTI_QUERY_MIN_ITEMS = 1
        mock_settings.ORCH_SEMANTIC_PLANNER = False
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False
        mock_settings.ORCH_DETERMINISTIC_SUBQUERY_SEMANTIC_TAIL = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))
        mock_contract_client.multi_query = AsyncMock(
            return_value={
                "items": [{"content": "c1"}, {"content": "c2"}],
                "trace": {},
                "subqueries": [],
            }
        )

        with patch(
            "app.infrastructure.clients.http_adapters.build_deterministic_subqueries",
            return_value=[{"id": "s1", "query": "sub1"}],
        ) as mock_builder:
            results = await retriever.retrieve_chunks(
                query="consulta multinorma",
                tenant_id="t1",
                collection_id="c1",
                plan=plan,
            )

        assert len(results) == 2
        assert mock_builder.call_args.kwargs["include_semantic_tail"] is False
        assert (
            retriever.last_retrieval_diagnostics.trace["deterministic_subquery_semantic_tail"]
            is False
        )


@pytest.mark.asyncio
async def test_retrieve_advanced_evaluator_override(retriever, mock_contract_client):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = True
        mock_settings.ORCH_MULTI_QUERY_MIN_ITEMS = 5
        mock_settings.ORCH_MULTI_QUERY_EVALUATOR = True
        mock_settings.ORCH_SEMANTIC_PLANNER = False
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))

        # Primary returns only 2 items (less than min 5)
        mock_contract_client.multi_query = AsyncMock(
            return_value={"items": [{"content": "c1"}, {"content": "c2"}]}
        )

        with patch("app.infrastructure.clients.http_adapters.RetrievalSufficiencyEvaluator") as mock_eval_class:
            mock_evaluator = MagicMock()
            mock_eval_class.return_value = mock_evaluator
            mock_evaluator.evaluate = AsyncMock(
                return_value=MagicMock(sufficient=True, reason="Evaluator says OK")
            )

            results = await retriever.retrieve_chunks(
                query="q", tenant_id="t", collection_id=None, plan=plan
            )

            assert len(results) == 2
            assert retriever.last_retrieval_diagnostics.strategy == "multi_query_primary_evaluator"
            assert retriever.last_retrieval_diagnostics.trace["evaluator_override"] is True


@pytest.mark.asyncio
async def test_retrieve_advanced_refinement_cycle(retriever, mock_contract_client):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = True
        mock_settings.ORCH_MULTI_QUERY_MIN_ITEMS = 5
        mock_settings.ORCH_MULTI_QUERY_REFINE = True
        mock_settings.ORCH_MULTI_QUERY_EVALUATOR = False
        mock_settings.ORCH_SEMANTIC_PLANNER = False
        mock_settings.ORCH_PLANNER_MAX_QUERIES = 5
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))

        # 1st call insufficient, 2nd call (refine) sufficient
        mock_contract_client.multi_query = AsyncMock(
            side_effect=[
                {"items": [{"content": "p1"}]},  # primary
                {
                    "items": [
                        {"content": "p1"},
                        {"content": "r1"},
                        {"content": "r2"},
                        {"content": "r3"},
                        {"content": "r4"},
                    ]
                },  # refine
            ]
        )

        # Mock build_deterministic_subqueries to return few items so step_back isn't sliced off
        with patch(
            "app.infrastructure.clients.http_adapters.build_deterministic_subqueries",
            return_value=[{"id": "s1", "query": "sub1"}],
        ):
            results = await retriever.retrieve_chunks(
                query="q", tenant_id="t", collection_id=None, plan=plan
            )

            assert len(results) == 5
            assert retriever.last_retrieval_diagnostics.strategy == "multi_query_refined"
            assert "multi_query_refine" in retriever.last_retrieval_diagnostics.trace["timings_ms"]
            assert mock_contract_client.multi_query.call_count == 2

            # Verify refinement call included step_back subquery
            call2_args = mock_contract_client.multi_query.call_args_list[1]
            queries_sent = call2_args.kwargs["queries"]
            assert any(q["id"] == "step_back" for q in queries_sent)


@pytest.mark.asyncio
async def test_retrieve_advanced_fallback_to_hybrid(retriever, mock_contract_client):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = True
        mock_settings.ORCH_MULTI_QUERY_MIN_ITEMS = 5
        mock_settings.ORCH_MULTI_QUERY_REFINE = False
        mock_settings.ORCH_MULTI_QUERY_EVALUATOR = False
        mock_settings.ORCH_MULTIHOP_FALLBACK = False
        mock_settings.ORCH_SEMANTIC_PLANNER = False
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))

        # Multi-query insufficient
        mock_contract_client.multi_query = AsyncMock(
            return_value={"items": [{"content": "too few"}]}
        )
        # Hybrid called as fallback
        mock_contract_client.hybrid = AsyncMock(
            return_value={"items": [{"content": "hybrid result"}], "trace": {"h": "t"}}
        )

        results = await retriever.retrieve_chunks(
            query="q", tenant_id="t", collection_id=None, plan=plan
        )

        assert len(results) == 1
        assert results[0].content == "hybrid result"
        assert retriever.last_retrieval_diagnostics.strategy == "hybrid"


@pytest.mark.asyncio
async def test_retrieve_advanced_legacy_multihop_fallback(retriever, mock_contract_client):
    with patch("app.infrastructure.clients.http_adapters.settings") as mock_settings:
        mock_settings.ORCH_RETRIEVAL_CONTRACT = "advanced"
        mock_settings.ORCH_MULTI_QUERY_PRIMARY = False  # Primary MQ disabled
        mock_settings.ORCH_MULTIHOP_FALLBACK = True
        mock_settings.ORCH_SEMANTIC_PLANNER = False
        mock_settings.ORCH_COVERAGE_GATE_ENABLED = False

        plan = create_retrieval_plan(standards=("ISO 9001", "ISO 14001"))

        # Hybrid returns something that triggers fallback
        mock_contract_client.hybrid = AsyncMock(
            return_value={"items": [{"content": "h1", "metadata": {"row": {}}}]}
        )

        with patch("app.infrastructure.clients.http_adapters.decide_multihop_fallback") as mock_decide:
            mock_decide.return_value = MagicMock(needs_fallback=True, reason="Legacy fallback")

            mock_contract_client.multi_query = AsyncMock(
                return_value={"items": [{"content": "mq items"}]}
            )

            results = await retriever.retrieve_chunks(
                query="q", tenant_id="t", collection_id=None, plan=plan
            )

            assert retriever.last_retrieval_diagnostics.strategy == "multi_query"
            assert (
                retriever.last_retrieval_diagnostics.trace["fallback_reason"] == "Legacy fallback"
            )
