import argparse
import asyncio
from dataclasses import dataclass

from app.ui.cli import chat_cli_runtime
from sdk.python.cire_rag_sdk import TenantContext


class _Client:
    pass


@dataclass
class _Tenant:
    id: str
    name: str


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        orchestrator_url="http://localhost:8001",
        doctor_query="q",
    )


def test_doctor_reports_no_token_and_no_tenants(monkeypatch, capsys):
    async def _tenants(*args, **kwargs):
        return []

    monkeypatch.setattr(chat_cli_runtime, "list_authorized_tenants", _tenants)

    asyncio.run(
        chat_cli_runtime._run_doctor(
            client=_Client(),  # type: ignore[arg-type]
            args=_args(),
            tenant_context=TenantContext(),
            access_token="",
            collection_id=None,
        )
    )
    output = capsys.readouterr().out
    assert "auth_ok: no" in output
    assert "tenant_count: 0" in output
    assert "retrieval_probe: skipped" in output


def test_doctor_reports_tenant_without_collections(monkeypatch, capsys):
    async def _tenants(*args, **kwargs):
        return [_Tenant(id="t1", name="T1")]

    async def _collections(*args, **kwargs):
        return []

    async def _post_answer(**kwargs):
        return {
            "mode": "literal_normativa",
            "context_chunks": [],
            "citations": [],
            "validation": {"accepted": True},
        }

    monkeypatch.setattr(chat_cli_runtime, "list_authorized_tenants", _tenants)
    monkeypatch.setattr(chat_cli_runtime, "list_authorized_collections", _collections)
    monkeypatch.setattr(chat_cli_runtime, "_post_answer", _post_answer)

    ctx = TenantContext(tenant_id="t1")
    asyncio.run(
        chat_cli_runtime._run_doctor(
            client=_Client(),  # type: ignore[arg-type]
            args=_args(),
            tenant_context=ctx,
            access_token="token",
            collection_id=None,
        )
    )
    output = capsys.readouterr().out
    assert "selected_tenant: t1" in output
    assert "collection_count: 0" in output
    assert "context_chunks_count: 0" in output


def test_rewrite_query_with_dynamic_mode_identifier() -> None:
    out = chat_cli_runtime._rewrite_query_with_clarification(
        "pregunta original",
        "cross_standard_analysis",
    )
    assert "__clarified_mode__=cross_standard_analysis" in out


def test_rewrite_query_with_scope_clarification_text() -> None:
    out = chat_cli_runtime._rewrite_query_with_clarification(
        "pregunta original",
        "Cobertura completa",
    )
    assert "__clarified_scope__=true" in out
    assert "__coverage__=full" in out
