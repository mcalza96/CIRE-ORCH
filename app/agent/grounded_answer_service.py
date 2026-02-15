from __future__ import annotations

from openai import AsyncOpenAI
import structlog

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
        mode: str = "explicativa",
        require_literal_evidence: bool = False,
        max_chunks: int = 10,
    ) -> str:
        if not query.strip():
            return "No hay una pregunta para responder."

        if not context_chunks:
            return "No tengo informacion suficiente en el contexto para responder."

        context = "\n\n".join(context_chunks[: max(1, max_chunks)]).strip()
        if not self._client:
            return context_chunks[0][:500]

        # "comparativa" is often interpretive. Only force strict literal formatting when explicitly required.
        strict = bool(require_literal_evidence) or mode in {"literal_normativa", "literal_lista"}
        if strict:
            system = (
                "Eres un auditor de normas ISO. Responde SOLO con evidencia del CONTEXTO. "
                "Cada afirmacion clave debe incluir al menos una cita con marcadores exactos "
                "del tipo C# o R# tal como aparecen en el CONTEXTO (ej: C1, R2). "
                "Si no hay evidencia suficiente, responde: "
                "\"No encontrado explicitamente en el contexto recuperado. Referencias revisadas: C1, C2, ...\""
            )
            style = (
                "FORMATO:\n"
                "1) Para cada afirmacion clave: Clausula | Cita literal breve | Fuente(C# o R#).\n"
                "2) No inventes clausulas ni texto normativo.\n"
            )
        else:
            system = (
                "Responde solo con evidencia del CONTEXTO. "
                "Puedes interpretar y conectar requisitos entre normas usando evidencias separadas "
                "(no tiene que existir una frase que las conecte literalmente), "
                "pero cada afirmacion relevante debe incluir al menos una referencia C# o R# "
                "tal como aparecen en el CONTEXTO (ej: C3, R1). "
                "Si una parte no esta respaldada por evidencia, marca: "
                "\"No encontrado explicitamente en el contexto recuperado\" y no inventes."
            )
            style = (
                "Estructura recomendada:\n"
                "- ISO 45001 8.1.2: que exige y como aplica al cambio.\n"
                "- ISO 14001 8.1: que reevaluar en ciclo de vida por el cambio.\n"
                "- ISO 9001 8.5.1: impacto documental/validacion de proceso.\n"
                "- Integracion: que documentos/registros cambian y por que.\n"
                "Incluye referencias (C#/R#) al final de cada punto."
            )

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
                        "content": (
                            f"PREGUNTA:\n{query}\n\n"
                            f"CONTEXTO:\n{context}\n\n"
                            f"{style}\n"
                        ),
                    },
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or "No tengo informacion suficiente en el contexto para responder."
        except Exception as exc:
            logger.warning("grounded_answer_model_fallback", error=str(exc))
            # Fallback defensivo: no bloquear el flujo por fallas de proveedor/modelo.
            return context_chunks[0][:500]
