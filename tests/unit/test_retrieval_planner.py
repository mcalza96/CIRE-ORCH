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
    subqueries = build_deterministic_subqueries(
        query=query, requested_standards=requested, max_queries=6
    )
    assert 1 <= len(subqueries) <= 6
    ids = {item["id"] for item in subqueries}
    # Agnostic bridge queries should be present when there is room.
    assert "bridge_contexto" in ids or "step_back" in ids


def test_build_deterministic_subqueries_literal_mode_defers_step_back() -> None:
    query = "ISO 9001 7.5.3 ISO 14001 7.4 ISO 45001 5.4 textualmente"
    requested = ("ISO 9001", "ISO 14001", "ISO 45001")
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=6,
        mode="literal_normativa",
    )
    ids = {item["id"] for item in subqueries}
    assert "step_back" not in ids


def test_build_deterministic_subqueries_semantic_tail_preserves_keywords_without_clause() -> None:
    query = (
        "Que exige textualmente la ISO 9001 sobre control de la informacion documentada, "
        "la ISO 14001 sobre comunicacion documentada y la ISO 45001 sobre participacion y consulta"
    )
    requested = ("ISO 9001", "ISO 14001", "ISO 45001")
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=6,
        mode="literal_normativa",
        include_semantic_tail=True,
    )
    scope_queries = [item for item in subqueries if str(item.get("id", "")).startswith("scope_")]
    assert len(scope_queries) == 3
    for item in scope_queries:
        text = str(item.get("query") or "").lower()
        assert "iso " in text
        assert len(text) <= 900
        assert any(
            token in text
            for token in (
                "informacion documentada",
                "control",
                "comunicacion documentada",
                "participacion",
                "consulta",
            )
        )


def test_build_deterministic_subqueries_semantic_tail_keeps_clause_plus_keywords() -> None:
    query = "ISO 9001 7.5.3 control de la informacion documentada textualmente con citas"
    requested = ("ISO 9001",)
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=4,
        mode="literal_normativa",
        include_semantic_tail=True,
    )
    assert subqueries
    qtext = str(subqueries[0]["query"]).lower()
    assert "iso 9001" in qtext
    assert "7.5.3" in qtext
    assert "control" in qtext
    assert "informacion documentada" in qtext


def test_build_deterministic_subqueries_semantic_tail_disabled() -> None:
    query = "Que exige textualmente la ISO 9001 sobre control documental"
    requested = ("ISO 9001",)
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=3,
        mode="literal_normativa",
        include_semantic_tail=False,
    )
    assert subqueries
    assert str(subqueries[0]["query"]).strip() == "ISO 9001"
