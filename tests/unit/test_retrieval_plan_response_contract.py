from app.agent.models import QueryIntent
from app.agent.policies import build_retrieval_plan
from app.cartridges.models import AgentProfile, QueryModeConfig, QueryModesPolicy, RetrievalPolicy


def test_build_retrieval_plan_propagates_mode_inference_and_response_contract() -> None:
    profile = AgentProfile(
        profile_id="iso",
        query_modes=QueryModesPolicy(
            default_mode="grounded_inference",
            modes={
                "grounded_inference": QueryModeConfig(
                    allow_inference=True,
                    require_literal_evidence=False,
                    retrieval_profile="explanatory_audit",
                    response_contract="grounded_inference",
                )
            },
        ),
        retrieval=RetrievalPolicy(
            by_mode={
                "explanatory_audit": {
                    "chunk_k": 30,
                    "chunk_fetch_k": 120,
                    "summary_k": 5,
                    "require_literal_evidence": False,
                }
            }
        ),
    )

    plan = build_retrieval_plan(
        QueryIntent(mode="grounded_inference"),
        query="analiza riesgo operativo",
        profile=profile,
    )

    assert plan.allow_inference is True
    assert plan.response_contract == "grounded_inference"
    assert plan.require_literal_evidence is False
