from app.agent.retrieval.retrieval_planner import (
    apply_search_hints,
    build_deterministic_subqueries,
    extract_clause_refs,
)
from app.profiles.models import AgentProfile, RouterHeuristics, SearchHint


def test_extract_clause_refs_uses_profile_reference_patterns() -> None:
    profile = AgentProfile(
        profile_id="legal-custom",
        router=RouterHeuristics(reference_patterns=[r"\bart\.?\s*\d+(?:\s*letra\s*[a-z])?\b"]),
    )

    refs = extract_clause_refs("Analiza art. 5 letra c del codigo", profile=profile)
    assert refs == ["art. 5 letra c"]


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
