from app.agent.policies import extract_requested_scopes, suggest_scope_candidates
from app.cartridges.models import AgentProfile, RouterHeuristics, ScopePattern


def test_suggest_scope_candidates_uses_profile_scope_hints() -> None:
    profile = AgentProfile(
        profile_id="custom",
        router=RouterHeuristics(
            scope_hints={
                "Codigo Civil": ["civil", "obligaciones"],
                "Codigo del Trabajo": ["laboral", "trabajador"],
            }
        ),
    )

    options = suggest_scope_candidates("analisis laboral con foco en trabajador", profile=profile)
    assert options[0] == "Codigo del Trabajo"


def test_extract_requested_scopes_uses_profile_scope_patterns() -> None:
    profile = AgentProfile(
        profile_id="iso-like",
        router=RouterHeuristics(
            scope_patterns=[
                ScopePattern(label="STD 100", regex=r"\bstd\s*100\b|\b100\b"),
                ScopePattern(label="STD 200", regex=r"\bstd\s*200\b|\b200\b"),
            ]
        ),
    )

    scopes = extract_requested_scopes("comparar STD 100 y STD 200", profile=profile)
    assert scopes == ("STD 100", "STD 200")


def test_extract_requested_scopes_falls_back_to_iso_references() -> None:
    profile = AgentProfile(
        profile_id="base-like",
        domain_entities=["requisito", "evidencia"],
    )

    scopes = extract_requested_scopes(
        "Relaciona ISO 9001:2015 con ISO 14001:2015 en este cambio",
        profile=profile,
    )
    assert scopes == ("ISO 9001", "ISO 14001")


def test_extract_requested_scopes_does_not_use_generic_domain_entities_as_scopes() -> None:
    profile = AgentProfile(
        profile_id="base-like",
        domain_entities=["requisito", "evidencia", "fuente"],
    )

    scopes = extract_requested_scopes(
        "que requisito documental aplica para este caso",
        profile=profile,
    )
    assert scopes == ()
