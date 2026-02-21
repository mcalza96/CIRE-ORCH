import pytest

from app.agent.components.grounded_answer_service import GroundedAnswerService
from app.profiles.models import AgentProfile, ValidationPolicy


@pytest.mark.asyncio
async def test_service_uses_profile_fallback_when_context_empty() -> None:
    service = GroundedAnswerService()
    profile = AgentProfile(
        profile_id="policy-profile",
        validation=ValidationPolicy(
            require_citations=True,
            forbidden_concepts=[],
            fallback_message="Fallback de perfil por evidencia insuficiente.",
        ),
    )

    answer = await service.generate_answer(
        query="Pregunta sin evidencia",
        context_chunks=[],
        agent_profile=profile,
    )

    assert answer == "Fallback de perfil por evidencia insuficiente."
