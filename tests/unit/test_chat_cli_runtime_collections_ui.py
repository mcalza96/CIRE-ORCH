import argparse
import asyncio

from app import chat_cli_runtime
from app.core.orch_discovery_client import Collection


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        collection_id=None,
        collection_name=None,
        non_interactive=False,
        orchestrator_url="http://localhost:8001",
    )


def test_print_answer_shows_empty_retrieval_hint(capsys):
    chat_cli_runtime._print_answer(
        {
            "answer": "No tengo informacion suficiente en el contexto para responder.",
            "mode": "explicativa",
            "citations": [],
            "context_chunks": [],
            "validation": {"accepted": True},
        }
    )
    out = capsys.readouterr().out
    assert "Sin evidencia recuperada" in out


def test_resolve_collection_displays_id_and_key(monkeypatch, capsys):
    async def _collections(*args, **kwargs):
        return [
            Collection(id="c1", name="ISO", collection_key="iso"),
            Collection(id="c2", name="SM 24 - micro", collection_key=None),
        ]

    monkeypatch.setattr(chat_cli_runtime, "list_authorized_collections", _collections)
    monkeypatch.setattr(chat_cli_runtime, "_prompt", lambda _msg: "0")

    collection_id, collection_name = asyncio.run(
        chat_cli_runtime._resolve_collection(
            args=_args(),
            tenant_id="t1",
            access_token="token",
        )
    )
    assert collection_id is None
    assert collection_name is None
    out = capsys.readouterr().out
    assert "id=c1" in out
    assert "key=iso" in out
