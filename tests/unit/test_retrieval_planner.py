from app.agent.retrieval_planner import decide_multihop_fallback, build_deterministic_subqueries


def test_decide_multihop_fallback_when_missing_standard_and_planner_not_multihop() -> None:
    query = "ISO 45001 8.1.2 vs ISO 14001 8.1 impacto documental ISO 9001 8.5.1"
    requested = ("ISO 45001", "ISO 14001", "ISO 9001")
    # Only one standard appears in top rows.
    rows = [
        {"content": "texto 8.1.2", "metadata": {"source_standard": "ISO 45001"}},
        {"content": "texto 8.1.2", "metadata": {"source_standard": "ISO 45001"}},
    ]
    decision = decide_multihop_fallback(
        query=query,
        requested_standards=requested,
        items=rows,
        hybrid_trace={"planner_multihop": False},
        top_k=12,
    )
    assert decision.needs_fallback is True


def test_build_deterministic_subqueries_bounded() -> None:
    query = "ISO 45001:2018 (Cláusula 8.1.2) ISO 14001:2015 (Cláusula 8.1) ISO 9001:2015 (Cláusula 8.5.1)"
    requested = ("ISO 45001", "ISO 14001", "ISO 9001")
    subqueries = build_deterministic_subqueries(query=query, requested_standards=requested, max_queries=6)
    assert 1 <= len(subqueries) <= 6
    ids = {item["id"] for item in subqueries}
    assert any("impacto_documental" in item_id for item_id in ids)

