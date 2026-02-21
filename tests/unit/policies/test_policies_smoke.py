from app.agent.policies.scope_policy import ScopePolicy
from app.agent.policies.retry_policy import RetryPolicy
from app.agent.models import RetrievalPlan, EvidenceItem, QueryIntent
from app.profiles.models import AgentProfile, QueryModeConfig, QueryModesPolicy
from dataclasses import dataclass


def test_scope_policy_missing():
    policy = ScopePolicy()
    plan = RetrievalPlan(
        mode="literal_normativa",
        chunk_k=1,
        chunk_fetch_k=1,
        summary_k=1,
        require_literal_evidence=True,
        requested_standards=("ISO 9001", "ISO 45001"),
    )

    # Mock evidence with metadata
    @dataclass
    class MockEvidence:
        metadata: dict
        content: str = ""

    evidence = [MockEvidence(metadata={"row": {"source_standard": "ISO 9001:2015"}})]

    missing = policy.detect_missing_scopes(evidence, plan)  # type: ignore
    assert "ISO 45001" in missing
    assert "ISO 9001" not in missing


def test_retry_policy_relax():
    policy = RetryPolicy()
    plan = RetrievalPlan(
        mode="literal_normativa",
        chunk_k=1,
        chunk_fetch_k=1,
        summary_k=1,
        require_literal_evidence=True,
        requested_standards=("ISO 9001", "ISO 45001"),
    )

    next_intent = policy.determine_next_intent(plan, "scope_mismatch")
    assert next_intent is not None
    assert next_intent.mode == "default"


def test_retry_policy_relax_uses_profile_dynamic_mode() -> None:
    policy = RetryPolicy()
    plan = RetrievalPlan(
        mode="literal_clause_check",
        chunk_k=1,
        chunk_fetch_k=1,
        summary_k=1,
        require_literal_evidence=True,
        requested_standards=("ISO 9001", "ISO 45001"),
    )
    profile = AgentProfile(
        profile_id="p",
        query_modes=QueryModesPolicy(
            default_mode="explanatory",
            modes={
                "literal_clause_check": QueryModeConfig(require_literal_evidence=True),
                "cross_scope_analysis": QueryModeConfig(
                    require_literal_evidence=False,
                    tool_hints=["logical_comparison"],
                ),
                "explanatory": QueryModeConfig(require_literal_evidence=False),
            },
        ),
    )

    next_intent = policy.determine_next_intent(plan, "scope_mismatch", profile=profile)
    assert next_intent is not None
    assert next_intent.mode == "cross_scope_analysis"
