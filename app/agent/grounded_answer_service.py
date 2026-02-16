from __future__ import annotations

from openai import AsyncOpenAI
import structlog

from app.cartridges.models import AgentProfile
from app.core.config import settings

logger = structlog.get_logger(__name__)


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
        mode: str = "explicativa",
        require_literal_evidence: bool = False,
        max_chunks: int = 10,
    ) -> str:
        profile_fallback = (
            str(agent_profile.validation.fallback_message).strip()
            if agent_profile is not None and agent_profile.validation.fallback_message
            else "No tengo informacion suficiente en el contexto para responder."
        )
        if not query.strip():
            return "No hay una pregunta para responder."

        if not context_chunks:
            return profile_fallback

        context = "\n\n".join(context_chunks[: max(1, max_chunks)]).strip()
        if not self._client:
            return context_chunks[0][:500]

        # "comparativa" is often interpretive. Only force strict literal formatting when explicitly required.
        strict = bool(require_literal_evidence) or mode in {"literal_normativa", "literal_lista"}
        synthesis = agent_profile.synthesis if agent_profile is not None else None
        identity = agent_profile.identity if agent_profile is not None else None
        identity_lines: list[str] = []
        if identity is not None:
            if identity.role:
                identity_lines.append(f"Rol operativo: {identity.role}")
            if identity.tone:
                identity_lines.append(f"Tono: {identity.tone}")
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
            else "C#/R#"
        )
        strict_subject_label = (
            str(synthesis.strict_subject_label).strip()
            if synthesis is not None and synthesis.strict_subject_label
            else "Afirmacion"
        )
        strict_reference_label = (
            str(synthesis.strict_reference_label).strip()
            if synthesis is not None and synthesis.strict_reference_label
            else "Referencia"
        )

        ref_label = synthesis.strict_reference_label if synthesis else "Evidencia"
        subj_label = synthesis.strict_subject_label if synthesis else "Afirmacion"
        cite_fmt = synthesis.citation_format if synthesis else "C#/R#"
        style_guide_text = "\n".join(
            f"- {rule}" for rule in (identity.style_guide if identity is not None else [])
        ).strip()

        system_prompt_base = (
            f"{persona_prefix}\n\n"
            "REGLAS DE SINTESIS:\n"
            + "\n".join(f"- {r}" for r in (synthesis.synthesis_rules if synthesis is not None else []))
            + f"\n- Utiliza el formato de cita: {cite_fmt}."
            + f"\n- Cada {subj_label} debe estar vinculada a su {ref_label}."
        )

        if strict:
            strict_system = (
                "Responde SOLO con evidencia del CONTEXTO. "
                "Cada afirmacion clave debe incluir al menos una cita con marcadores exactos "
                f"con el formato '{cite_fmt}' tal como aparecen en el CONTEXTO. "
                "Si no hay evidencia suficiente, responde: "
                '"No encontrado explicitamente en el contexto recuperado. Referencias revisadas: <lista>"'
            )
            style = (
                "FORMATO:\n"
                f"1) Para cada afirmacion clave: {strict_subject_label} | Cita literal breve | {strict_reference_label}.\n"
                "2) No inventes referencias ni texto fuente.\n"
            )
            system = f"{persona_prefix}\n\n{strict_system}".strip()
            if rules_text:
                system = f"{system}\n\nReglas:\n{rules_text}".strip()
            if style_strict:
                style = f"{style}\n{style_strict}".strip()
            if style_guide_text:
                style = f"{style}\n\nGuia de estilo:\n{style_guide_text}".strip()
        else:
            interpretive_system = (
                "Responde solo con evidencia del CONTEXTO. "
                "Puedes interpretar y conectar requisitos entre fuentes usando evidencias separadas "
                "(no tiene que existir una frase que las conecte literalmente), "
                "pero cada afirmacion relevante debe incluir al menos una referencia C# o R# "
                "tal como aparecen en el CONTEXTO (ej: C3, R1). "
                "Si una parte no esta respaldada por evidencia, marca: "
                '"No encontrado explicitamente en el contexto recuperado" y no inventes.'
            )
            style = (
                "Estructura recomendada:\n"
                "- Hallazgo principal respaldado por evidencia.\n"
                "- Implicaciones operativas y tecnicas.\n"
                "- Riesgos o vacios de evidencia detectados.\n"
                "- Integracion entre fuentes y conclusion.\n"
                "Incluye referencias (C#/R#) al final de cada punto."
            )
            system = f"{persona_prefix}\n\n{interpretive_system}".strip()
            if rules_text:
                system = f"{system}\n\nReglas:\n{rules_text}".strip()
            if style_interpretive:
                style = f"{style}\n{style_interpretive}".strip()
            if style_guide_text:
                style = f"{style}\n\nGuia de estilo:\n{style_guide_text}".strip()

        style = f"{style}\n\nFormato de cita requerido: {citation_format}".strip()

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
                        "content": (f"PREGUNTA:\n{query}\n\nCONTEXTO:\n{context}\n\n{style}\n"),
                    },
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or profile_fallback
        except Exception as exc:
            logger.warning("grounded_answer_model_fallback", error=str(exc))
            # Fallback defensivo: no bloquear el flujo por fallas de proveedor/modelo.
            return context_chunks[0][:500]
