from app.cartridges.models import AgentProfile, CapabilitiesPolicy, RouterHeuristics
from app.graph.nodes.universal_planner import build_universal_plan


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
