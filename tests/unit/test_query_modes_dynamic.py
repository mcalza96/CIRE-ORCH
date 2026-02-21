from app.agent.policies import build_retrieval_plan, classify_intent
from app.agent.models import QueryIntent
from app.profiles.models import (
    AgentProfile,
    QueryModeConfig,
    QueryModesPolicy,
    RetrievalModeConfig,
    RetrievalPolicy,
    IntentRule,
)


def _dynamic_profile() -> AgentProfile:
    return AgentProfile(
        profile_id="dynamic",
        query_modes=QueryModesPolicy(
            default_mode="general_answer",
            modes={
                "triage": QueryModeConfig(
                    require_literal_evidence=True,
                    allow_inference=False,
                    retrieval_profile="literal_normativa",
                ),
                "general_answer": QueryModeConfig(
                    require_literal_evidence=False,
                    allow_inference=True,
                    retrieval_profile="explicativa",
                ),
            },
            intent_rules=[
                IntentRule(
                    id="triage_by_symptoms",
                    mode="triage",
                    any_keywords=["fiebre", "dolor"],
                )
            ],
        ),
        retrieval=RetrievalPolicy(
            by_mode={
                "literal_normativa": RetrievalModeConfig(
                    chunk_k=40,
                    chunk_fetch_k=160,
                    summary_k=2,
                    require_literal_evidence=True,
                ),
                "explicativa": RetrievalModeConfig(
                    chunk_k=20,
                    chunk_fetch_k=100,
                    summary_k=4,
                    require_literal_evidence=False,
                ),
            }
        ),
    )


def test_classify_intent_uses_profile_query_modes_rules() -> None:
    profile = _dynamic_profile()
    intent = classify_intent("Paciente con fiebre y dolor de cabeza", profile=profile)
    assert intent.mode == "triage"
    assert "profile_rule" in intent.rationale


def test_classify_intent_uses_profile_default_mode_when_no_rule_matches() -> None:
    profile = _dynamic_profile()
    intent = classify_intent("Necesito una explicacion general", profile=profile)
    assert intent.mode == "general_answer"


def test_build_retrieval_plan_resolves_retrieval_profile_from_dynamic_mode() -> None:
    profile = _dynamic_profile()
    plan = build_retrieval_plan(
        QueryIntent(mode="triage", rationale="rule"),
        query="Paciente con fiebre",
        profile=profile,
    )
    assert plan.mode == "triage"
    assert plan.chunk_k == 40
    assert plan.chunk_fetch_k == 160
    assert plan.summary_k == 2
    assert plan.require_literal_evidence is True
