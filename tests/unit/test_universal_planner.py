from app.profiles.models import (
    AgentProfile,
    CapabilitiesPolicy,
    IntentRule,
    QueryModeConfig,
    QueryModesPolicy,
    RouterHeuristics,
)
from app.graph.universal.planning import build_universal_plan


def test_universal_planner_simple_query_uses_single_retrieval_step() -> None:
    profile = AgentProfile(
        profile_id="p",
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=["semantic_retrieval", "citation_validator"],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="Que exige ISO 9001 en 9.1?",
        profile=profile,
        allowed_tools=["semantic_retrieval", "citation_validator"],
    )
    assert plan.steps
    assert plan.steps[0].tool == "semantic_retrieval"


def test_universal_planner_never_uses_tool_outside_allowlist() -> None:
    profile = AgentProfile(
        profile_id="p",
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=["semantic_retrieval"],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="Calcula 5*(20+2) para 20 muestras de arsenico",
        profile=profile,
        allowed_tools=["semantic_retrieval"],
    )
    assert {step.tool for step in plan.steps} <= {"semantic_retrieval"}


def test_universal_planner_uses_profile_calculation_patterns() -> None:
    profile = AgentProfile(
        profile_id="p",
        router=RouterHeuristics(calculation_patterns=[r"\bcompute\b"]),
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=["semantic_retrieval", "python_calculator"],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="Please compute total batch amount",
        profile=profile,
        allowed_tools=["semantic_retrieval", "python_calculator"],
    )
    assert {step.tool for step in plan.steps} >= {"semantic_retrieval", "python_calculator"}


def test_universal_planner_does_not_use_language_hardcoded_patterns() -> None:
    profile = AgentProfile(
        profile_id="p",
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=["semantic_retrieval", "python_calculator"],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="calcula lote de muestra",
        profile=profile,
        allowed_tools=["semantic_retrieval", "python_calculator"],
    )
    assert {step.tool for step in plan.steps} == {"semantic_retrieval"}


def test_universal_planner_respects_mode_execution_plan_order() -> None:
    profile = AgentProfile(
        profile_id="p",
        query_modes=QueryModesPolicy(
            default_mode="ops_mode",
            modes={
                "ops_mode": QueryModeConfig(
                    execution_plan=[
                        "semantic_retrieval",
                        "structural_extraction",
                        "citation_validator",
                    ],
                    retrieval_profile="explanatory_response",
                )
            },
            intent_rules=[IntentRule(id="ops", mode="ops_mode", any_keywords=["ops"])],
        ),
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=[
                "semantic_retrieval",
                "structural_extraction",
                "citation_validator",
            ],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="ops check",
        profile=profile,
        allowed_tools=["semantic_retrieval", "structural_extraction", "citation_validator"],
    )
    assert [step.tool for step in plan.steps] == [
        "semantic_retrieval",
        "structural_extraction",
        "citation_validator",
    ]


def test_universal_planner_supports_expectation_coverage_tool() -> None:
    profile = AgentProfile(
        profile_id="p",
        query_modes=QueryModesPolicy(
            default_mode="gap_analysis",
            modes={
                "gap_analysis": QueryModeConfig(
                    execution_plan=[
                        "semantic_retrieval",
                        "expectation_coverage",
                        "citation_validator",
                    ],
                    retrieval_profile="explanatory_response",
                )
            },
            intent_rules=[
                IntentRule(id="gap", mode="gap_analysis", any_keywords=["gap", "brecha"])
            ],
        ),
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=[
                "semantic_retrieval",
                "expectation_coverage",
                "citation_validator",
            ],
        ),
    )
    _, _, plan, _ = build_universal_plan(
        query="analiza brecha de evidencia",
        profile=profile,
        allowed_tools=["semantic_retrieval", "expectation_coverage", "citation_validator"],
    )
    assert [step.tool for step in plan.steps] == [
        "semantic_retrieval",
        "expectation_coverage",
        "citation_validator",
    ]
