from __future__ import annotations

import argparse
import glob
import os
from typing import Any
from uuid import uuid4

from app.infrastructure.clients.auth_client import ensure_access_token
from app.ui.ingestion_models import IngestionRuntime
from app.ui.ingestion_monitoring import run_batch_monitoring, run_job_monitoring
from app.ui.ingestion_selection import (
    choose_menu_option,
    collect_files_interactively,
    pick_documents_for_replay,
    prompt,
    resolve_collection,
    resolve_operation,
    resolve_tenant,
)
from sdk.python.cire_rag_sdk.client import AsyncCireRagClient


async def resolve_operation_for_run(args: argparse.Namespace) -> str:
    operation = resolve_operation(args)
    if operation != "menu":
        return operation
    if args.non_interactive:
        raise RuntimeError(
            "No action provided. Use --file/--glob, --resume-batch o --replay-enrichment-doc."
        )
    return choose_menu_option()


async def build_runtime(args: argparse.Namespace, operation: str) -> IngestionRuntime:
    access_token = args.access_token or await ensure_access_token(
        interactive=not args.non_interactive
    )
    tenant_id = await resolve_tenant(args, access_token)
    return IngestionRuntime(
        args=args,
        operation=operation,
        access_token=access_token,
        tenant_id=tenant_id,
    )


def build_default_headers(*, tenant_id: str, access_token: str) -> dict[str, str]:
    return {
        "X-Tenant-ID": tenant_id,
        "Authorization": f"Bearer {access_token}",
        "X-Correlation-ID": str(uuid4()),
        "X-Request-ID": str(uuid4()),
    }


async def run_selected_operation(*, client: AsyncCireRagClient, runtime: IngestionRuntime) -> None:
    if runtime.operation == "resume":
        await _run_resume_operation(client=client, runtime=runtime)
        return
    if runtime.operation == "replay":
        await _run_replay_operation(client=client, runtime=runtime)
        return
    await _run_ingest_operation(client=client, runtime=runtime)


def _resolve_batch_id_for_resume(runtime: IngestionRuntime) -> str:
    batch_id = str(runtime.args.resume_batch or "").strip()
    if batch_id:
        return batch_id
    if runtime.args.non_interactive:
        raise RuntimeError("--resume-batch is required in non-interactive mode")
    prompted = prompt("📝 Batch ID a monitorear: ")
    if prompted:
        return prompted
    raise RuntimeError("Batch ID requerido")


async def _run_resume_operation(*, client: AsyncCireRagClient, runtime: IngestionRuntime) -> None:
    batch_id = _resolve_batch_id_for_resume(runtime)
    await run_batch_monitoring(client, batch_id, poll_seconds=runtime.args.job_poll_seconds)


async def _resolve_replay_doc_ids(
    *, client: AsyncCireRagClient, runtime: IngestionRuntime
) -> list[str]:
    doc_id = str(runtime.args.replay_enrichment_doc or "").strip()
    if doc_id:
        return [doc_id]
    if runtime.args.non_interactive:
        raise RuntimeError("--replay-enrichment-doc is required in non-interactive mode")
    doc_ids = await pick_documents_for_replay(
        args=runtime.args,
        client=client,
        tenant_id=runtime.tenant_id,
        access_token=runtime.access_token,
    )
    if doc_ids:
        return doc_ids
    raise RuntimeError("No hay documentos para replay")


async def _run_replay_operation(*, client: AsyncCireRagClient, runtime: IngestionRuntime) -> None:
    doc_ids = await _resolve_replay_doc_ids(client=client, runtime=runtime)

    replay_single_doc = len(doc_ids) == 1 and bool(
        str(runtime.args.replay_enrichment_doc or "").strip()
    )
    if replay_single_doc:
        include_visual = not runtime.args.replay_no_visual
        include_graph = not runtime.args.replay_no_graph
        include_raptor = not runtime.args.replay_no_raptor
    else:
        # Replay interactivo por colección: SOLO GRAFOS por defecto.
        include_visual = False
        include_graph = True
        include_raptor = False

    queued_jobs: list[tuple[str, str]] = []
    for doc_id in doc_ids:
        res = await client.replay_enrichment(
            doc_id,
            include_visual=include_visual,
            include_graph=include_graph,
            include_raptor=include_raptor,
            tenant_id=runtime.tenant_id,
        )
        job_id = str(res.get("job_id") or "").strip() if isinstance(res, dict) else ""
        queued_jobs.append((doc_id, job_id))
        print(f"✅ Replay encolado para doc={doc_id} | job_id={job_id or 'n/a'}")

    print(
        f"🧾 Replay batch encolado: {len(queued_jobs)} docs | visual={include_visual} graph={include_graph} raptor={include_raptor}"
    )

    if runtime.args.no_wait:
        return

    for _, job_id in queued_jobs:
        if job_id:
            await run_job_monitoring(client, job_id, poll_seconds=runtime.args.job_poll_seconds)


def _collect_files_from_args(args: argparse.Namespace) -> list[str]:
    files_to_upload: list[str] = []
    if args.file:
        files_to_upload.extend(args.file)
    if args.glob:
        for pattern in args.glob:
            files_to_upload.extend(glob.glob(pattern))
    return files_to_upload


def _resolve_source_standard(args: argparse.Namespace) -> str:
    if args.non_interactive:
        return ""
    ans = prompt("📝 Norma o estandar fuente (ej. ISO 9001) [Enter para ignorar/adivinar]: ")
    return ans.strip().upper()


async def _upload_files_to_batch(
    *,
    client: AsyncCireRagClient,
    batch_id: str,
    files_to_upload: list[str],
    source_standard: str,
) -> None:
    for file_path in files_to_upload:
        print(f"📤 Subiendo {os.path.basename(file_path)}...")
        file_meta: dict[str, Any] = {}
        if source_standard:
            file_meta["source_standard"] = source_standard
        await client.upload_file_to_batch(
            batch_id,
            file_path,
            metadata=file_meta if file_meta else None,
        )


async def _run_ingest_operation(*, client: AsyncCireRagClient, runtime: IngestionRuntime) -> None:
    collection_id = await resolve_collection(runtime.args, runtime.tenant_id, runtime.access_token)
    files_to_upload = _collect_files_from_args(runtime.args)
    if not files_to_upload:
        if runtime.args.non_interactive:
            print("❌ No hay archivos para subir.")
            return
        files_to_upload.extend(collect_files_interactively())

    print(f"📦 Creando batch para {len(files_to_upload)} archivos...")
    batch = await client.create_ingestion_batch(
        tenant_id=runtime.tenant_id,
        collection_id=collection_id,
        collection_key=collection_id,
        collection_name=collection_id,
        total_files=len(files_to_upload),
        auto_seal=False,
        embedding_mode=runtime.args.embedding_mode,
    )
    batch_id = str(batch.get("id") or batch.get("batch_id") or "").strip()
    if not batch_id:
        raise RuntimeError(f"create_ingestion_batch no devolvio id/batch_id. Respuesta: {batch}")
    print(f"✅ Batch creado: {batch_id}")

    source_standard = _resolve_source_standard(runtime.args)
    await _upload_files_to_batch(
        client=client,
        batch_id=batch_id,
        files_to_upload=files_to_upload,
        source_standard=source_standard,
    )

    print("🔗 Sellando batch...")
    await client.seal_ingestion_batch(batch_id)
    print("✅ Batch sellado.")
    if not runtime.args.no_wait:
        await run_batch_monitoring(client, batch_id, poll_seconds=runtime.args.job_poll_seconds)
