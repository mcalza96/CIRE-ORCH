import pytest

from app.profiles.models import AgentProfile


def test_capabilities_defaults_are_safe() -> None:
    profile = AgentProfile(profile_id="base-test")
    assert profile.capabilities.reasoning_level == "low"
    assert profile.capabilities.reasoning_budget.max_steps == 4
    assert profile.capabilities.reasoning_budget.max_reflections == 2
    assert "semantic_retrieval" in profile.capabilities.allowed_tools
    assert "citation_validator" in profile.capabilities.allowed_tools
    assert profile.retrieval.by_mode == {}


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


def test_capabilities_accepts_dynamic_tool_names() -> None:
    profile = AgentProfile.model_validate(
        {
            "profile_id": "dyn-tools",
            "capabilities": {
                "allowed_tools": ["semantic_retrieval", "external_sql"],
                "tool_policies": {
                    "external_sql": {
                        "enabled": True,
                        "max_input_chars": 2048,
                    }
                },
            },
        }
    )
    assert "external_sql" in profile.capabilities.allowed_tools
    assert profile.capabilities.tool_policies["external_sql"].enabled is True


def test_retrieval_mode_config_enforces_hard_limits() -> None:
    with pytest.raises(Exception):
        AgentProfile.model_validate(
            {
                "profile_id": "bad-profile",
                "retrieval": {
                    "by_mode": {
                        "m": {
                            "chunk_k": 999,
                            "chunk_fetch_k": 999,
                            "summary_k": 99,
                        }
                    }
                },
            }
        )
