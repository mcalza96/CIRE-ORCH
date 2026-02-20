from app.agent.retrieval_planner import (
    build_deterministic_subqueries,
    normalize_query_filters,
)


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
    scope_filters = {
        str((item.get("filters") or {}).get("source_standard") or "").upper()
        for item in subqueries
        if isinstance(item, dict)
    }
    assert {"ISO 45001", "ISO 14001", "ISO 9001"}.issubset(scope_filters)


def test_build_deterministic_subqueries_literal_mode_defers_step_back() -> None:
    query = "ISO 9001 7.5.3 ISO 14001 7.4 ISO 45001 5.4 textualmente"
    requested = ("ISO 9001", "ISO 14001", "ISO 45001")
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=6,
        mode="literal_normativa",
        require_literal_evidence=True,
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
        require_literal_evidence=True,
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
        require_literal_evidence=True,
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
        require_literal_evidence=True,
        include_semantic_tail=False,
    )
    assert subqueries
    assert str(subqueries[0]["query"]).strip() == "ISO 9001"


def test_build_deterministic_subqueries_adds_clause_focused_queries_for_dense_query() -> None:
    query = "ISO 9001 5.3 analiza impacto cruzado en 10.3 y 9.1.2"
    requested = ("ISO 9001",)
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=8,
        require_literal_evidence=False,
    )
    clause_ids = [
        str(item.get("id") or "")
        for item in subqueries
        if str(item.get("id") or "").startswith("clause_")
    ]
    assert clause_ids
    assert any("10_3" in cid for cid in clause_ids)
    assert any("9_1_2" in cid for cid in clause_ids)


def test_normalize_query_filters_enforces_single_standard_selector() -> None:
    raw = {
        "source_standard": "ISO 9001",
        "source_standards": ["ISO 9001", "ISO 14001"],
        "filters": {"clause_id": "8.1.2"},
    }
    normalized = normalize_query_filters(raw)
    assert normalized is not None
    assert normalized.get("source_standard") == "ISO 9001"
    assert "source_standards" not in normalized
    assert normalized.get("metadata", {}).get("clause_id") == "8.1.2"


def test_clause_focused_subquery_avoids_global_clause_filter_when_ambiguous() -> None:
    query = "compara requisitos de 8.1.2 entre ISO 9001 e ISO 14001"
    requested = ("ISO 9001", "ISO 14001")
    subqueries = build_deterministic_subqueries(
        query=query,
        requested_standards=requested,
        max_queries=8,
    )
    clause_queries = [
        item for item in subqueries if str(item.get("id") or "").startswith("clause_")
    ]
    assert clause_queries
    for item in clause_queries:
        filters = item.get("filters") or {}
        assert isinstance(filters, dict)
        assert not ("source_standard" in filters and "source_standards" in filters)
        # Ambiguous cross-standard clause should not hard-lock metadata clause_id globally.
        assert "metadata" not in filters
