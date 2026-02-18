import asyncio

from app.agent.grounded_answer_service import GroundedAnswerService
from app.cartridges.models import AgentProfile, IdentityPolicy, SynthesisPolicy, ValidationPolicy


class _FakeCompletionMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeCompletionMessage(content)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("_Resp", (), {"choices": [_FakeChoice("ok")]})()


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def test_grounded_answer_service_uses_profile_templates_and_not_mode_name() -> None:
    service = GroundedAnswerService()
    fake_completions = _FakeCompletions()
    service._client = _FakeClient(fake_completions)  # type: ignore[attr-defined]

    profile = AgentProfile(
        profile_id="p",
        identity=IdentityPolicy(role="Auditor", tone="Neutral"),
        validation=ValidationPolicy(fallback_message="fallback"),
        synthesis=SynthesisPolicy(
            citation_format="REF#",
            strict_subject_label="Finding",
            strict_reference_label="Source",
            identity_role_prefix="Role",
            identity_tone_prefix="Tone",
            strict_system_prompt_template="STRICT::{citation_format}",
            interpretive_system_prompt_template="INTERPRET::{citation_format}",
            strict_style_template="STRICT_STYLE::{strict_subject_label}|{strict_reference_label}",
            interpretive_style_template="INTERP_STYLE::{citation_format}",
            user_prompt_template="Q={query}\nCTX={context}\nSTYLE={style}",
        ),
    )

    result = asyncio.run(
        service.generate_answer(
            query="question",
            context_chunks=["ctx1"],
            agent_profile=profile,
            mode="literal_normativa",
            require_literal_evidence=False,
        )
    )

    assert result == "ok"
    assert fake_completions.calls
    messages = fake_completions.calls[0]["messages"]
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    assert "INTERPRET::REF#" in system_message
    assert "STRICT::" not in system_message
    assert "Role: Auditor" in system_message
    assert "Tone: Neutral" in system_message
    assert "INTERP_STYLE::REF#" in user_message
