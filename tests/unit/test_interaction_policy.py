from app.agent.models import QueryIntent, ReasoningPlan, RetrievalPlan, ToolCall
from app.agent.policies import extract_requested_scopes
from app.cartridges.models import AgentProfile
from app.graph.universal.interaction import decide_interaction


def _profile() -> AgentProfile:
    data = {
        "profile_id": "base",
        "interaction_policy": {
            "enabled": True,
            "max_interruptions_per_turn": 1,
            "thresholds": {
                "l2_ambiguity": 0.35,
                "l3_subqueries": 6,
                "l3_latency_s": 15,
                "l3_cost_tokens": 12000,
                "low_coverage": 0.5,
            },
            "mode_overrides": {
                "cross_scope_analysis": {
                    "require_plan_approval": False,
                    "risk_level": "medium",
                    "required_slots": ["scope"],
                },
                "gap_analysis": {
                    "require_plan_approval": True,
                    "risk_level": "high",
                    "required_slots": ["scope", "objective"],
                },
            },
        },
        "scope_resolution": {
            "canonical_scopes": ["ISO 9001", "ISO 14001", "ISO 45001"],
            "aliases": {
                "ISO 9001": ["9001", "iso 90001", "gestion de la calidad"],
                "ISO 14001": ["14001", "gestion ambiental"],
                "ISO 45001": ["45001", "sst"],
            },
        },
        "query_modes": {
            "default_mode": "explanatory_response",
            "modes": {
                "explanatory_response": {"decomposition_policy": {"max_subqueries": 2}},
                "cross_scope_analysis": {"decomposition_policy": {"max_subqueries": 8}},
                "gap_analysis": {"decomposition_policy": {"max_subqueries": 12}},
            },
        },
    }
    return AgentProfile.model_validate(data)


def test_extract_requested_scopes_handles_typo_alias() -> None:
    scopes = extract_requested_scopes("Que exige la iso 90001?", profile=_profile())
    assert "ISO 9001" in scopes


def test_extract_requested_scopes_reads_requested_scopes_marker() -> None:
    scopes = extract_requested_scopes(
        "pregunta\n\n__requested_scopes__=[ISO 9001|ISO 14001]",
        profile=_profile(),
    )
    assert "ISO 9001" in scopes
    assert "ISO 14001" in scopes


def test_decide_interaction_l2_for_ambiguous_scope() -> None:
    profile = _profile()
    decision = decide_interaction(
        query="Compara las normas iso",
        intent=QueryIntent(mode="cross_scope_analysis"),
        retrieval_plan=RetrievalPlan(
            mode="cross_scope_analysis",
            chunk_k=30,
            chunk_fetch_k=120,
            summary_k=5,
            requested_standards=(),
        ),
        reasoning_plan=ReasoningPlan(goal="x", steps=[ToolCall(tool="semantic_retrieval")]),
        profile=profile,
        prior_interruptions=0,
    )
    assert decision.needs_interrupt is True
    assert decision.level == "L2"
    assert decision.kind == "clarification"


def test_decide_interaction_l3_for_complex_high_risk_mode() -> None:
    profile = _profile()
    decision = decide_interaction(
        query="haz un gap analysis completo entre 9001, 14001 y 45001",
        intent=QueryIntent(mode="gap_analysis"),
        retrieval_plan=RetrievalPlan(
            mode="gap_analysis",
            chunk_k=45,
            chunk_fetch_k=220,
            summary_k=8,
            requested_standards=("ISO 9001", "ISO 14001", "ISO 45001"),
        ),
        reasoning_plan=ReasoningPlan(
            goal="x",
            steps=[
                ToolCall(tool="semantic_retrieval"),
                ToolCall(tool="logical_comparison"),
                ToolCall(tool="expectation_coverage"),
            ],
            complexity="complex",
        ),
        profile=profile,
        prior_interruptions=0,
    )
    assert decision.needs_interrupt is True
    assert decision.level == "L3"
    assert decision.kind == "plan_approval"


def test_decide_interaction_respects_structured_clarification_context() -> None:
    profile = _profile()
    decision = decide_interaction(
        query="Compara las normas iso",
        intent=QueryIntent(mode="cross_scope_analysis"),
        retrieval_plan=RetrievalPlan(
            mode="cross_scope_analysis",
            chunk_k=30,
            chunk_fetch_k=120,
            summary_k=5,
            requested_standards=("ISO 9001", "ISO 14001"),
        ),
        reasoning_plan=ReasoningPlan(goal="x", steps=[ToolCall(tool="semantic_retrieval")]),
        profile=profile,
        prior_interruptions=0,
        clarification_context={
            "round": 1,
            "kind": "clarification",
            "selected_option": "si, continuar",
            "confirmed": True,
            "requested_scopes": ["ISO 9001", "ISO 14001"],
            "answer_text": "Comparar por objetivo de riesgo operativo",
        },
    )
    assert decision.needs_interrupt is False
    assert decision.level == "L1"


def test_decide_interaction_cross_scope_prompt_uses_generic_example_without_iso() -> None:
    profile = _profile()
    decision = decide_interaction(
        query="compara los enfoques del libro",
        intent=QueryIntent(mode="cross_scope_analysis"),
        retrieval_plan=RetrievalPlan(
            mode="cross_scope_analysis",
            chunk_k=30,
            chunk_fetch_k=120,
            summary_k=5,
            requested_standards=(),
        ),
        reasoning_plan=ReasoningPlan(goal="x", steps=[ToolCall(tool="semantic_retrieval")]),
        profile=profile,
        prior_interruptions=0,
    )
    assert decision.needs_interrupt is True
    assert decision.level == "L2"
    assert "ISO 9001" not in decision.question
