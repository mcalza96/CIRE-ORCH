from __future__ import annotations

from openai import AsyncOpenAI
import structlog

from app.cartridges.models import AgentProfile
from app.core.config import settings

logger = structlog.get_logger(__name__)


_DEFAULT_FALLBACK_MESSAGE = "Insufficient context evidence to answer."
_DEFAULT_EMPTY_QUERY_MESSAGE = "No question provided."
_DEFAULT_CITATION_FORMAT = "C#/R#"
_DEFAULT_STRICT_SYSTEM_TEMPLATE = (
    "Answer only with evidence from CONTEXT. Every key claim must include at least one "
    "exact citation using '{citation_format}'. If evidence is missing, state that clearly."
)
_DEFAULT_INTERPRETIVE_SYSTEM_TEMPLATE = (
    "Answer only with evidence from CONTEXT. You may connect evidence across sources, "
    "but mark unsupported parts explicitly and do not invent references."
)
_DEFAULT_STRICT_STYLE_TEMPLATE = (
    "FORMAT:\n"
    "1) For each key claim: {strict_subject_label} | short literal quote | {strict_reference_label}.\n"
    "2) Do not invent references or source text."
)
_DEFAULT_INTERPRETIVE_STYLE_TEMPLATE = (
    "Recommended structure:\n"
    "- Main finding backed by evidence.\n"
    "- Operational implications.\n"
    "- Evidence gaps and risks.\n"
    "- Integrated conclusion."
)
_DEFAULT_IDENTITY_ROLE_PREFIX = "Role"
_DEFAULT_IDENTITY_TONE_PREFIX = "Tone"
_DEFAULT_USER_PROMPT_TEMPLATE = "QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n{style}\n"


class GroundedAnswerService:
    def __init__(self) -> None:
        self._client = (
            AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            if settings.GROQ_API_KEY
            else None
        )

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[str],
        *,
        agent_profile: AgentProfile | None = None,
        mode: str = "default",
        require_literal_evidence: bool = False,
        max_chunks: int = 10,
    ) -> str:
        profile_fallback = (
            str(agent_profile.validation.fallback_message).strip()
            if agent_profile is not None and agent_profile.validation.fallback_message
            else _DEFAULT_FALLBACK_MESSAGE
        )
        if not query.strip():
            return _DEFAULT_EMPTY_QUERY_MESSAGE

        if not context_chunks:
            return profile_fallback

        context = "\n\n".join(context_chunks[: max(1, max_chunks)]).strip()
        if not self._client:
            return context_chunks[0][:500]

        del mode
        strict = bool(require_literal_evidence)
        synthesis = agent_profile.synthesis if agent_profile is not None else None
        identity = agent_profile.identity if agent_profile is not None else None
        role_prefix = (
            str(synthesis.identity_role_prefix).strip()
            if synthesis is not None and synthesis.identity_role_prefix
            else _DEFAULT_IDENTITY_ROLE_PREFIX
        )
        tone_prefix = (
            str(synthesis.identity_tone_prefix).strip()
            if synthesis is not None and synthesis.identity_tone_prefix
            else _DEFAULT_IDENTITY_TONE_PREFIX
        )
        identity_lines: list[str] = []
        if identity is not None:
            if identity.role:
                identity_lines.append(f"{role_prefix}: {identity.role}")
            if identity.tone:
                identity_lines.append(f"{tone_prefix}: {identity.tone}")
        persona_prefix = "\n".join(
            line
            for line in [
                "\n".join(identity_lines).strip(),
                str(synthesis.system_persona).strip()
                if synthesis is not None and synthesis.system_persona
                else "",
            ]
            if line
        ).strip()
        rules_text = "\n".join(
            f"- {rule}" for rule in (synthesis.synthesis_rules if synthesis is not None else [])
        )
        style_strict = "\n".join(synthesis.strict_style) if synthesis is not None else ""
        style_interpretive = (
            "\n".join(synthesis.interpretive_style) if synthesis is not None else ""
        )
        citation_format = (
            str(synthesis.citation_format).strip()
            if synthesis is not None and synthesis.citation_format
            else _DEFAULT_CITATION_FORMAT
        )
        strict_subject_label = (
            str(synthesis.strict_subject_label).strip()
            if synthesis is not None and synthesis.strict_subject_label
            else "Claim"
        )
        strict_reference_label = (
            str(synthesis.strict_reference_label).strip()
            if synthesis is not None and synthesis.strict_reference_label
            else "Reference"
        )
        style_guide_text = "\n".join(
            f"- {rule}" for rule in (identity.style_guide if identity is not None else [])
        ).strip()

        strict_system_template = (
            str(synthesis.strict_system_prompt_template).strip()
            if synthesis is not None and synthesis.strict_system_prompt_template
            else _DEFAULT_STRICT_SYSTEM_TEMPLATE
        )
        interpretive_system_template = (
            str(synthesis.interpretive_system_prompt_template).strip()
            if synthesis is not None and synthesis.interpretive_system_prompt_template
            else _DEFAULT_INTERPRETIVE_SYSTEM_TEMPLATE
        )
        strict_style_template = (
            str(synthesis.strict_style_template).strip()
            if synthesis is not None and synthesis.strict_style_template
            else _DEFAULT_STRICT_STYLE_TEMPLATE
        )
        interpretive_style_template = (
            str(synthesis.interpretive_style_template).strip()
            if synthesis is not None and synthesis.interpretive_style_template
            else _DEFAULT_INTERPRETIVE_STYLE_TEMPLATE
        )

        template_data = {
            "citation_format": citation_format,
            "strict_subject_label": strict_subject_label,
            "strict_reference_label": strict_reference_label,
        }

        if strict:
            strict_system = strict_system_template.format(**template_data)
            style = strict_style_template.format(**template_data)
            system = f"{persona_prefix}\n\n{strict_system}".strip()
            if rules_text:
                system = f"{system}\n\nRules:\n{rules_text}".strip()
            if style_strict:
                style = f"{style}\n{style_strict}".strip()
            if style_guide_text:
                style = f"{style}\n\nStyle guide:\n{style_guide_text}".strip()
        else:
            interpretive_system = interpretive_system_template.format(**template_data)
            style = interpretive_style_template.format(**template_data)
            system = f"{persona_prefix}\n\n{interpretive_system}".strip()
            if rules_text:
                system = f"{system}\n\nRules:\n{rules_text}".strip()
            if style_interpretive:
                style = f"{style}\n{style_interpretive}".strip()
            if style_guide_text:
                style = f"{style}\n\nStyle guide:\n{style_guide_text}".strip()

        style = f"{style}\n\nRequired citation format: {citation_format}".strip()
        user_prompt_template = (
            str(synthesis.user_prompt_template).strip()
            if synthesis is not None and synthesis.user_prompt_template
            else _DEFAULT_USER_PROMPT_TEMPLATE
        )
        user_prompt = user_prompt_template.format(query=query, context=context, style=style)

        try:
            completion = await self._client.chat.completions.create(
                model=settings.GROQ_MODEL_CHAT,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or profile_fallback
        except Exception as exc:
            logger.warning("grounded_answer_model_fallback", error=str(exc))
            # Fallback defensivo: no bloquear el flujo por fallas de proveedor/modelo.
            return context_chunks[0][:500]
