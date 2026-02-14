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

    async def generate_answer(self, query: str, context_chunks: list[str], max_chunks: int = 10) -> str:
        if not query.strip():
            return "No hay una pregunta para responder."

        if not context_chunks:
            return "No tengo informacion suficiente en el contexto para responder."

        context = "\n\n".join(context_chunks[: max(1, max_chunks)]).strip()
        if not self._client:
            return context_chunks[0][:500]

        try:
            completion = await self._client.chat.completions.create(
                model=settings.GROQ_MODEL_CHAT,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Responde solo con evidencia del contexto. "
                            "Si no hay evidencia suficiente, dilo explicitamente."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"PREGUNTA:\n{query}\n\n"
                            f"CONTEXTO:\n{context}\n\n"
                            "Respuesta breve en espanol."
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
