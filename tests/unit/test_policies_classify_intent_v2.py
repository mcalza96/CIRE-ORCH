from app.agent.policies import classify_intent_with_trace
from app.cartridges.models import AgentProfile, IntentRule, QueryModeConfig, QueryModesPolicy


def test_classify_intent_with_trace_uses_profile_rule_mode() -> None:
    profile = AgentProfile(
        profile_id="p",
        query_modes=QueryModesPolicy(
            default_mode="general",
            modes={
                "cross_scope": QueryModeConfig(retrieval_profile="comparativa"),
                "general": QueryModeConfig(retrieval_profile="explicativa"),
            },
            intent_rules=[
                IntentRule(
                    id="rule_compare", mode="cross_scope", any_keywords=["impacta", "compara"]
                )
            ],
        ),
    )
    query = (
        "Analice cómo la falta de responsabilidad (Cláusula 5.3) impacta la participación (ISO 45001 5.4) "
        "y cómo impide la mejora continua (10.3) basándose en datos de satisfacción (ISO 9001 9.1.2) "
        "y desempeño ambiental (ISO 14001 9.1.1)."
    )
    intent, trace = classify_intent_with_trace(query, profile=profile)
    assert intent.mode == "cross_scope"
    assert isinstance(trace, dict)
    assert trace.get("version") == "profile_rules_v1"


def test_classify_intent_with_trace_low_signal_defaults_to_profile_default_mode() -> None:
    profile = AgentProfile(
        profile_id="iso_like",
        query_modes=QueryModesPolicy(
            default_mode="general",
            modes={
                "general": QueryModeConfig(retrieval_profile="explicativa"),
            },
        ),
    )
    intent, trace = classify_intent_with_trace(
        "que dice la introduccion de la iso 9001?",
        profile=profile,
    )
    assert intent.mode == "general"
    assert trace.get("version") == "profile_rules_v1"
