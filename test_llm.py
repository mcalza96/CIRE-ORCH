import asyncio
import json
from openai import AsyncOpenAI
import os

_SYSTEM_PROMPT = (
    "Eres un asistente de aclaracion para un orquestador RAG. "
    "DEBES devolver UNICAMENTE un objeto JSON válido con este exacto esquema:\n"
    '{"question": "string", "options": ["string"], "missing_slots": ["string"], '
    '"expected_answer": "string", "confidence": 0.9}\n'
    "No devuelvas ningún texto antes ni después del JSON. No uses bloques markdown."
)

async def test_groq():
    from app.core.config import settings
    api_key = getattr(settings, "GROQ_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = getattr(settings, "ORCH_CLARIFICATION_MODEL", "llama-3.1-8b-instant")
    prompt = {
        "query": "compara las normas iso",
        "current_question": "heuristic",
        "current_options": ["op1"],
        "missing_slots": ["objective", "scope"],
        "scope_candidates": ["ISO 9001", "ISO 14001"],
        "instructions": "Redacta pregunta pidiendo scope y objective."
    }
    
    try:
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant", # or whatever groq model is configured
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "Genera el JSON para este contexto:\n" + json.dumps(prompt)}
            ],
            response_format={"type": "json_object"}
        )
        print("Success:", completion.choices[0].message.content)
    except Exception as e:
        print("Error:", e)
        
if __name__ == "__main__":
    asyncio.run(test_groq())
