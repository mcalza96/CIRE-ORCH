import argparse
import asyncio

from app.ui import renderers
from app.infrastructure import discovery_utils
from app.infrastructure.clients.discovery_client import Collection


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        collection_id=None,
        collection_name=None,
        non_interactive=False,
        orchestrator_url="http://localhost:8001",
    )


def test_print_answer_shows_empty_retrieval_hint(capsys):
    renderers.print_answer(
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

    monkeypatch.setattr(discovery_utils, "list_authorized_collections", _collections)
    monkeypatch.setattr(discovery_utils, "_prompt", lambda _msg: "0")

    collection_id, collection_name = asyncio.run(
        discovery_utils.resolve_collection(
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


def test_print_answer_diagnostics_maps_preflight_signature_warning(capsys):
    renderers.print_answer_diagnostics(
        {
            "context_chunks": [],
            "citations": [],
            "validation": {
                "accepted": False,
                "issues": ["Answer does not include explicit source markers (C#/R#)."],
            },
            "retrieval": {
                "contract": "advanced",
                "strategy": "hybrid",
                "trace": {
                    "warnings": ["hybrid_rpc_preflight_signature_mismatch"],
                    "warning_codes": ["HYBRID_RPC_SIGNATURE_MISMATCH_HNSW"],
                },
            },
            "retrieval_plan": {},
        }
    )
    out = capsys.readouterr().out
    assert "stage=rag_sql_contract" in out
    assert "desalineacion de firma RPC" in out


def test_print_answer_diagnostics_stream_missing_context_not_marked_retrieval(capsys):
    renderers.print_answer_diagnostics(
        {
            "citations": ["C5"],
            "validation": {
                "accepted": False,
                "issues": [
                    "Grounded inference requires at least 2 citations in Inferencias section."
                ],
            },
            "retrieval": {
                "contract": "advanced",
                "strategy": "langgraph_universal_flow",
                "trace": {},
            },
        }
    )
    out = capsys.readouterr().out
    assert "stage=validation" in out
    assert "no se recuperaron chunks para la respuesta" not in out


def test_print_answer_uses_context_chunks_count_in_stream_payload(capsys):
    renderers.print_answer(
        {
            "answer": "respuesta",
            "mode": "explicativa",
            "citations": [],
            "context_chunks_count": 0,
            "validation": {"accepted": True, "issues": []},
            "retrieval": {"contract": "advanced", "strategy": "langgraph_universal_flow"},
        }
    )
    out = capsys.readouterr().out
    assert "Sin evidencia recuperada" in out
    assert "stage=retrieval" in out
