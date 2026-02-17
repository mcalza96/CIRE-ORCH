from app.cartridges.models import AgentProfile


def test_capabilities_defaults_are_safe() -> None:
    profile = AgentProfile(profile_id="base-test")
    assert profile.capabilities.reasoning_level == "low"
    assert profile.capabilities.reasoning_budget.max_steps == 4
    assert profile.capabilities.reasoning_budget.max_reflections == 2
    assert "semantic_retrieval" in profile.capabilities.allowed_tools
    assert "citation_validator" in profile.capabilities.allowed_tools


def test_capabilities_accept_high_reasoning_payload() -> None:
    profile = AgentProfile.model_validate(
        {
            "profile_id": "lab-test",
            "capabilities": {
                "reasoning_level": "high",
                "allowed_tools": [
                    "semantic_retrieval",
                    "structural_extraction",
                    "python_calculator",
                    "citation_validator",
                ],
                "reasoning_budget": {"max_steps": 4, "max_reflections": 2},
                "tool_policies": {
                    "python_calculator": {
                        "enabled": True,
                        "max_expression_chars": 128,
                        "timeout_ms": 250,
                        "max_operations": 900,
                    }
                },
            },
        }
    )
    assert profile.capabilities.reasoning_level == "high"
    assert "python_calculator" in profile.capabilities.allowed_tools
    assert profile.capabilities.tool_policies["python_calculator"].enabled is True
