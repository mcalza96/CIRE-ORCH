"""CLI runtime for document ingestion using CireRagClient SDK."""

from __future__ import annotations

import argparse
import asyncio
import os

from app.ui.ingestion_operations import (
    build_default_headers,
    build_runtime,
    resolve_operation_for_run,
    run_selected_operation,
)
from sdk.python.cire_rag_sdk.client import AsyncCireRagClient, TenantContext


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document ingestion CLI via Orchestrator/RAG API")
    parser.add_argument("--tenant-id", help="Institutional tenant id")
    parser.add_argument("--tenant-name", help="Institutional tenant name (display)")
    parser.add_argument("--collection-id", help="Collection id/key")
    parser.add_argument("--collection-name", help="Collection name (display)")
    parser.add_argument("--file", action="append", help="File to upload (repeatable)")
    parser.add_argument("--glob", action="append", help="Glob pattern for files (repeatable)")
    parser.add_argument("--embedding-mode", choices=["LOCAL", "CLOUD"], help="Embedding mode")
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for processing to finish"
    )
    parser.add_argument(
        "--watch-mode",
        choices=["stream", "poll"],
        default="stream",
        help="Worker watch mode",
    )
    parser.add_argument("--resume-batch", help="Resume monitoring of an existing batch")
    parser.add_argument(
        "--replay-enrichment-doc", help="Replay enrichment for an existing document"
    )
    parser.add_argument(
        "--replay-no-visual",
        action="store_true",
        help="In replay, skip visual stage",
    )
    parser.add_argument(
        "--replay-no-graph",
        action="store_true",
        help="In replay, skip graph stage",
    )
    parser.add_argument(
        "--replay-no-raptor",
        action="store_true",
        help="In replay, skip raptor stage",
    )
    parser.add_argument(
        "--documents-limit",
        type=int,
        default=500,
        help="Document list limit for replay picker",
    )
    parser.add_argument(
        "--job-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval for status monitors",
    )
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ORCH_URL") or "http://localhost:8001",
        help="Orchestrator URL",
    )
    parser.add_argument(
        "--rag-url",
        default=os.getenv("RAG_URL") or "http://localhost:8000",
        help="RAG Engine URL",
    )
    parser.add_argument("--access-token", help="Bearer token for auth")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if user input required",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    operation = await resolve_operation_for_run(args)
    runtime = await build_runtime(args, operation)

    async with AsyncCireRagClient(
        base_url=args.rag_url,
        api_key=os.getenv("RAG_SERVICE_SECRET") or os.getenv("RAG_API_KEY"),
        default_headers=build_default_headers(
            tenant_id=runtime.tenant_id,
            access_token=runtime.access_token,
        ),
        tenant_context=TenantContext(tenant_id=runtime.tenant_id),
    ) as client:
        await run_selected_operation(client=client, runtime=runtime)


if __name__ == "__main__":
    asyncio.run(main())
