from app.agent.retrieval_planner import (
    apply_search_hints,
    build_deterministic_subqueries,
    build_initial_scope_filters,
    extract_clause_refs,
)
from app.cartridges.models import AgentProfile, RouterHeuristics, ScopePattern, SearchHint


def test_extract_clause_refs_uses_profile_reference_patterns() -> None:
    profile = AgentProfile(
        profile_id="legal-custom",
        router=RouterHeuristics(reference_patterns=[r"\bart\.?\s*\d+(?:\s*letra\s*[a-z])?\b"]),
    )

    refs = extract_clause_refs("Analiza art. 5 letra c del codigo", profile=profile)
    assert refs == ["art. 5 letra c"]


def test_build_initial_scope_filters_uses_first_extracted_reference() -> None:
    profile = AgentProfile(
        profile_id="test-iso",
        router=RouterHeuristics(
            scope_patterns=[
                ScopePattern(label="ISO 9001", regex=r"ISO 9001"),
                ScopePattern(label="ISO 14001", regex=r"ISO 14001"),
            ],
            reference_patterns=[r"\bart\.?\s*\d+(?:\s*letra\s*[a-z])?\b"],
        ),
    )
    filters = build_initial_scope_filters(
        plan_requested=(),
        mode="literal_normativa",
        query="cita art. 7 letra a",
        profile=profile,
        require_literal_evidence=True,
    )

    assert isinstance(filters, dict)
    metadata = filters.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("clause_id") == "art. 7 letra a"


def test_apply_search_hints_expands_query_terms() -> None:
    profile = AgentProfile(
        profile_id="test-hints",
    )
    profile.retrieval.search_hints = [
        SearchHint(term="epp", expand_to=["equipo de proteccion personal", "casco"])
    ]

    expanded, trace = apply_search_hints("revisar uso de epp", profile=profile)
    assert "equipo de proteccion personal" in expanded
    assert "casco" in expanded
    assert trace.get("applied")


def test_build_initial_scope_filters_does_not_force_clause_from_hint_expansion() -> None:
    profile = AgentProfile(profile_id="test-hints")
    profile.router.reference_patterns = [r"\b\d+(?:\.\d+)+\b"]
    profile.retrieval.search_hints = [
        SearchHint(term="introduccion", expand_to=["0.1", "objeto y campo de aplicacion"])
    ]

    filters = build_initial_scope_filters(
        plan_requested=("ISO 9001",),
        mode="literal_normativa",
        query="Que dice la introduccion de la ISO 9001",
        profile=profile,
        require_literal_evidence=True,
    )

    assert isinstance(filters, dict)
    metadata = filters.get("metadata")
    assert metadata is None


def test_build_deterministic_subqueries_ignores_hint_generated_clause_refs() -> None:
    profile = AgentProfile(profile_id="test-hints")
    profile.router.reference_patterns = [r"\b\d+(?:\.\d+)+\b"]
    profile.retrieval.search_hints = [SearchHint(term="introduccion", expand_to=["0.1", "0.2"])]

    subqueries = build_deterministic_subqueries(
        query="Que dice la introduccion de la ISO 9001",
        requested_standards=("ISO 9001",),
        mode="literal_normativa",
        require_literal_evidence=True,
        profile=profile,
    )

    assert subqueries
    scope_query = subqueries[0]
    metadata = scope_query.get("filters", {}).get("metadata", {})
    assert "clause_id" not in metadata


def test_build_initial_scope_filters_does_not_lock_clause_for_compound_query() -> None:
    profile = AgentProfile(profile_id="test-compound")
    profile.router.reference_patterns = [r"\b\d+(?:\.\d+)+\b"]

    filters = build_initial_scope_filters(
        plan_requested=("ISO 9001", "ISO 14001"),
        mode="literal_normativa",
        query=(
            "Que exige textualmente la clausula 9.3 de la ISO 9001? "
            "Enumera entradas y salidas esperadas de ISO 14001"
        ),
        profile=profile,
        require_literal_evidence=True,
    )

    assert isinstance(filters, dict)
    assert "metadata" not in filters


def test_policy_management_hint_expands_to_cross_standard_terms() -> None:
    profile = AgentProfile(profile_id="test-policy")
    profile.retrieval.search_hints = [
        SearchHint(
            term="politica de gestion",
            expand_to=["política de la calidad", "política ambiental", "política de sst"],
        )
    ]

    expanded, trace = apply_search_hints(
        "que exigen textualmente sobre politica de gestion", profile=profile
    )
    assert trace.get("applied") is True
    assert "política de la calidad" in expanded
    assert "política ambiental" in expanded
    assert "política de sst" in expanded
