"""HTTP-based chat CLI for split orchestrator architecture."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q/A chat via Orchestrator API")
    parser.add_argument("--tenant-id", required=True, help="Institutional tenant id")
    parser.add_argument("--collection-id", help="Collection id (optional)")
    parser.add_argument("--collection-name", help="Collection name (display only)")
    parser.add_argument(
        "--orchestrator-url",
        default="http://localhost:8001",
        help="Base URL for orchestrator API",
    )
    return parser.parse_args()


def _rewrite_query_with_clarification(original_query: str, clarification_answer: str) -> str:
    text = (clarification_answer or "").strip()
    if not text:
        return original_query
    return (
        f"{original_query}\n\n"
        "__clarified_scope__=true "
        f"Aclaracion de alcance: {text}."
    )


async def _post_answer(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_id: str,
    query: str,
    collection_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": tenant_id,
    }
    if collection_id:
        payload["collection_id"] = collection_id

    response = await client.post(orchestrator_url.rstrip("/") + "/api/v1/knowledge/answer", json=payload)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _print_answer(data: dict[str, Any]) -> None:
    answer = str(data.get("answer") or "").strip()
    mode = str(data.get("mode") or "").strip()
    citations = data.get("citations") if isinstance(data.get("citations"), list) else []
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    accepted = bool(validation.get("accepted", True))
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []

    print("\n" + "=" * 60)
    print(f"🤖 RESPUESTA ({mode or 'N/A'})")
    print("=" * 60)
    print(answer or "(sin respuesta)")
    if citations:
        print("\n📚 Citas: " + ", ".join(str(item) for item in citations[:10]))
    if not accepted and issues:
        print("⚠️ Validacion: " + "; ".join(str(issue) for issue in issues))
    print("=" * 60 + "\n")


async def main() -> None:
    args = parse_args()
    scope = args.collection_name or args.collection_id or "todo el tenant"
    print("🚀 Chat Q/A Orchestrator (split mode HTTP)")
    print(f"🏢 Tenant: {args.tenant_id}")
    print(f"📁 Scope: {scope}")
    print(f"🌐 Orchestrator URL: {args.orchestrator_url}")
    print("💡 Escribe tu pregunta (o 'salir')")

    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            query = input("❓ > ").strip()
            if query.lower() in {"salir", "exit", "quit"}:
                print("Okey, ¡adiós! 👋")
                return
            if not query:
                continue

            try:
                result = await _post_answer(
                    client,
                    args.orchestrator_url,
                    args.tenant_id,
                    query,
                    args.collection_id,
                )
                _print_answer(result)

                clarification = result.get("clarification") if isinstance(result.get("clarification"), dict) else None
                rounds = 0
                while clarification and rounds < 3:
                    question = str(clarification.get("question") or "").strip()
                    options = clarification.get("options") if isinstance(clarification.get("options"), list) else []
                    if question:
                        print("🧠 Clarificacion requerida: " + question)
                    if options:
                        print("🧩 Opciones: " + " | ".join(str(opt) for opt in options))
                    reply = input("📝 Aclaracion > ").strip()
                    if not reply:
                        break
                    clarified_query = _rewrite_query_with_clarification(query, reply)
                    result = await _post_answer(
                        client,
                        args.orchestrator_url,
                        args.tenant_id,
                        clarified_query,
                        args.collection_id,
                    )
                    _print_answer(result)
                    clarification = result.get("clarification") if isinstance(result.get("clarification"), dict) else None
                    rounds += 1
            except httpx.HTTPStatusError as exc:
                print(f"❌ Error HTTP {exc.response.status_code}: {exc.response.text}")
            except Exception as exc:
                print(f"❌ Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
