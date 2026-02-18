from app.agent.mode_classifier import classify_mode_v2
from app.cartridges.models import AgentProfile, IntentRule, QueryModeConfig, QueryModesPolicy


def test_mode_classifier_uses_profile_default_mode() -> None:
    profile = AgentProfile(
        profile_id="p",
        query_modes=QueryModesPolicy(
            default_mode="cross_scope",
            modes={
                "cross_scope": QueryModeConfig(retrieval_profile="comparativa"),
            },
        ),
    )
    query = (
        "Analice cómo la falta de definición de la responsabilidad y autoridad (Cláusula 5.3) "
        "impacta la consulta y participación (Cláusula 5.4) e impide demostrar la mejora continua (Cláusula 10.3). "
        "ISO 9001 9.1.2 e ISO 14001 9.1.1."
    )
    result = classify_mode_v2(query, profile=profile)
    assert result.mode == "cross_scope"
    assert result.blocked_modes == ()


def test_mode_classifier_returns_generic_default_without_profile() -> None:
    query = "Transcribe el texto exacto de la cláusula 8.4.3"
    result = classify_mode_v2(query)
    assert result.mode == "default"


def test_mode_classifier_uses_first_mode_when_default_missing() -> None:
    profile = AgentProfile(
        profile_id="iso_like",
        query_modes=QueryModesPolicy(
            default_mode="",
            modes={
                "triage": QueryModeConfig(retrieval_profile="literal_normativa"),
                "general": QueryModeConfig(retrieval_profile="explicativa"),
            },
            intent_rules=[IntentRule(id="unused", mode="triage", any_keywords=["fiebre"])],
        ),
    )
    query = "que dice la introduccion de la iso 9001?"
    result = classify_mode_v2(query, profile=profile)
    assert result.mode == "triage"
