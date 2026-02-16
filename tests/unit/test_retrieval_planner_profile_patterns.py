from app.agent.retrieval_planner import (
    apply_search_hints,
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
