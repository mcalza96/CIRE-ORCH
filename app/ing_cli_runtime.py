"""CLI runtime for document ingestion using CireRagClient SDK."""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.auth_client import ensure_access_token
from app.core.observability.ingestion_utils import human_ingestion_stage
from app.core.orch_discovery_client import list_authorized_collections, list_authorized_tenants
from sdk.python.cire_rag_sdk.client import AsyncCireRagClient, TenantContext

BATCH_TERMINAL_STATES = {"DONE", "ERROR", "CANCELLED"}
JOB_TERMINAL_STATES = {"completed", "failed", "cancelled", "error"}


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
        "--watch-mode", choices=["stream", "poll"], default="stream", help="Worker watch mode"
    )
    parser.add_argument("--resume-batch", help="Resume monitoring of an existing batch")
    parser.add_argument(
        "--replay-enrichment-doc", help="Replay enrichment for an existing document"
    )
    parser.add_argument(
        "--replay-no-visual", action="store_true", help="In replay, skip visual stage"
    )
    parser.add_argument(
        "--replay-no-graph", action="store_true", help="In replay, skip graph stage"
    )
    parser.add_argument(
        "--replay-no-raptor", action="store_true", help="In replay, skip raptor stage"
    )
    parser.add_argument(
        "--documents-limit", type=int, default=500, help="Document list limit for replay picker"
    )
    parser.add_argument(
        "--job-poll-seconds", type=float, default=5.0, help="Polling interval for status monitors"
    )
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ORCH_URL") or "http://localhost:8001",
        help="Orchestrator URL",
    )
    parser.add_argument(
        "--rag-url", default=os.getenv("RAG_URL") or "http://localhost:8000", help="RAG Engine URL"
    )
    parser.add_argument("--access-token", help="Bearer token for auth")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Fail if user input required"
    )
    return parser.parse_args(argv)


def _prompt(message: str) -> str:
    return input(message).strip()


async def resolve_tenant(args: argparse.Namespace, access_token: str) -> str:
    if args.tenant_id:
        return args.tenant_id

    try:
        tenants = await list_authorized_tenants(args.orchestrator_url, access_token)
    except Exception as exc:
        print(f"⚠️ Discovery falló: {exc}")
        tenants = []

    if not tenants:
        if args.non_interactive:
            raise RuntimeError("Tenant selection required")
        manual = _prompt("🏢 Tenant ID (UUID): ")
        if not manual:
            raise SystemExit("Tenant ID requerido.")
        return manual

    if len(tenants) == 1:
        print(f"🏢 Tenant auto-seleccionado: {tenants[0].name}")
        return tenants[0].id

    print("🏢 Tenants disponibles:")
    for idx, t in enumerate(tenants, start=1):
        print(f"  {idx}) {t.name} ({t.id})")

    while True:
        opt = _prompt(f"📝 Elige tenant [1-{len(tenants)}]: ")
        if opt.isdigit() and 1 <= int(opt) <= len(tenants):
            return tenants[int(opt) - 1].id
        print("Opción inválida.")


async def resolve_collection(args: argparse.Namespace, tenant_id: str, access_token: str) -> str:
    if args.collection_id:
        return args.collection_id

    try:
        cols = await list_authorized_collections(args.orchestrator_url, access_token, tenant_id)
    except Exception:
        cols = []

    if not cols:
        if args.non_interactive:
            return "default"
        manual = _prompt("📁 Collection Key (default): ")
        return manual or "default"

    print("📁 Colecciones disponibles:")
    for idx, c in enumerate(cols, start=1):
        print(f"  {idx}) {c.name} ({c.collection_key or c.id})")
    print("  0) Crear nueva")

    while True:
        opt = _prompt(f"📝 Elige colección [0-{len(cols)}]: ")
        if opt == "0":
            return _prompt("Nombre de la nueva colección: ")
        if opt.isdigit() and 1 <= int(opt) <= len(cols):
            col = cols[int(opt) - 1]
            return col.collection_key or col.id
        print("Opción inválida.")


def _resolve_operation(args: argparse.Namespace) -> str:
    if args.resume_batch:
        return "resume"
    if args.replay_enrichment_doc:
        return "replay"
    if args.file or args.glob:
        return "ingest"
    return "menu"


def _extract_documents(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


def _doc_label(doc: dict[str, Any]) -> str:
    filename = str(doc.get("filename") or "").strip()
    if filename:
        return filename
    source_path = str(doc.get("storage_path") or "").strip()
    if source_path:
        return Path(source_path).name
    return str(doc.get("id") or "(sin-id)")


def _status_timestamp(status: dict[str, Any]) -> str:
    updated = str(status.get("updated_at") or "").strip()
    if updated:
        return updated
    created = str(status.get("created_at") or "").strip()
    return created


def _choose_menu_option(args: argparse.Namespace) -> str:
    print("\n🧭 ¿Qué quieres hacer?")
    print("  1) Nueva ingesta")
    print("  2) Monitorear batch existente")
    print("  3) Replay de enrichment (sin re-ingesta)")
    while True:
        choice = _prompt("📝 Elige opción [1-3]: ")
        if choice == "1":
            return "ingest"
        if choice == "2":
            return "resume"
        if choice == "3":
            return "replay"
        print("Opción inválida.")


async def _pick_document_for_replay(
    *,
    args: argparse.Namespace,
    client: AsyncCireRagClient,
    tenant_id: str,
    access_token: str,
) -> str:
    docs_payload = await client.list_documents(limit=max(1, int(args.documents_limit)))
    all_docs = _extract_documents(docs_payload)
    if not all_docs:
        raise RuntimeError("No hay documentos disponibles para replay.")

    selected_docs = all_docs
    try:
        collections = await list_authorized_collections(
            args.orchestrator_url, access_token, tenant_id
        )
    except Exception as exc:
        print(f"⚠️ No se pudieron listar colecciones autorizadas: {exc}")
        collections = []

    if collections:
        print("📁 Colecciones disponibles:")
        for idx, collection in enumerate(collections, start=1):
            doc_count = sum(
                1
                for row in all_docs
                if str(row.get("collection_id") or "").strip() == str(collection.id).strip()
            )
            key = str(collection.collection_key or "").strip()
            suffix = f" | key={key}" if key else ""
            print(f"  {idx}) {collection.name} ({doc_count} docs){suffix}")

        while True:
            raw = _prompt(f"📝 Elige colección [1-{len(collections)}]: ")
            if raw.isdigit() and 1 <= int(raw) <= len(collections):
                picked = collections[int(raw) - 1]
                selected_docs = [
                    row
                    for row in all_docs
                    if str(row.get("collection_id") or "").strip() == str(picked.id).strip()
                ]
                if selected_docs:
                    break
                print("⚠️ Esta colección no tiene documentos en la lista actual.")
                continue
            print("Opción inválida.")

    if not selected_docs:
        raise RuntimeError("No hay documentos para la colección seleccionada.")

    print("📄 Documentos disponibles:")
    for idx, doc in enumerate(selected_docs, start=1):
        doc_id = str(doc.get("id") or "").strip()
        status = str(doc.get("status") or "unknown").strip().lower()
        created_at = str(doc.get("created_at") or "").strip()
        created_suffix = f" | {created_at}" if created_at else ""
        print(f"  {idx}) {_doc_label(doc)} [{status}] ({doc_id}){created_suffix}")

    while True:
        raw = _prompt(f"📝 Elige documento [1-{len(selected_docs)}] o pega doc_id: ")
        if raw.isdigit() and 1 <= int(raw) <= len(selected_docs):
            chosen = str(selected_docs[int(raw) - 1].get("id") or "").strip()
            if chosen:
                return chosen
        by_id = str(raw).strip()
        if by_id:
            return by_id
        print("Opción inválida.")


async def run_batch_monitoring(
    client: AsyncCireRagClient,
    batch_id: str,
    *,
    poll_seconds: float,
):
    print(f"🔎 Monitoreando batch {batch_id}...")
    last_snapshot = ""
    while True:
        try:
            status = await client.get_batch_status(batch_id)
            batch = status.get("batch", {})
            obs = status.get("observability", {})

            state = batch.get("status", "unknown")
            percent = obs.get("progress_percent", 0.0)
            stage = obs.get("dominant_stage", "OTHER")

            snapshot = f"{state}|{percent}|{stage}"
            if snapshot != last_snapshot:
                print(
                    f"📡 Estado: {state} | Progreso: {percent}% | Etapa: {human_ingestion_stage(stage)}"
                )
                last_snapshot = snapshot

            if str(state) in BATCH_TERMINAL_STATES:
                print(f"✅ Batch finalizado con estado: {state}")
                break
        except Exception as exc:
            print(f"⚠️ Error monitoreando: {exc}")

        await asyncio.sleep(max(1.0, float(poll_seconds)))


async def run_job_monitoring(
    client: AsyncCireRagClient,
    job_id: str,
    *,
    poll_seconds: float,
):
    print(f"🔎 Monitoreando job {job_id}...")
    last_snapshot = ""
    while True:
        try:
            status = await client.get_ingestion_job_status(job_id)
            state = str(status.get("status") or "unknown").strip().lower()
            job_type = str(status.get("job_type") or "unknown").strip()
            err = str(status.get("error_message") or "").strip()
            stamp = _status_timestamp(status)

            snapshot = f"{job_type}|{state}|{err}|{stamp}"
            if snapshot != last_snapshot:
                suffix = f" | error={err}" if err else ""
                stamp_suffix = f" | updated_at={stamp}" if stamp else ""
                print(f"📡 Job: {job_type} | Estado: {state}{stamp_suffix}{suffix}")
                last_snapshot = snapshot

            if state in JOB_TERMINAL_STATES:
                if state == "completed":
                    print("✅ Replay finalizado correctamente.")
                else:
                    print(f"❌ Replay terminó con estado: {state}")
                break
        except Exception as exc:
            print(f"⚠️ Error monitoreando job: {exc}")

        await asyncio.sleep(max(1.0, float(poll_seconds)))


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    access_token = args.access_token or await ensure_access_token(
        interactive=not args.non_interactive
    )

    operation = _resolve_operation(args)
    if operation == "menu":
        if args.non_interactive:
            raise RuntimeError(
                "No action provided. Use --file/--glob, --resume-batch o --replay-enrichment-doc."
            )
        operation = _choose_menu_option(args)

    tenant_id = await resolve_tenant(args, access_token)

    trace_id = str(uuid4())
    request_id = str(uuid4())
    default_headers = {
        "X-Tenant-ID": tenant_id,
        "Authorization": f"Bearer {access_token}",
        "X-Correlation-ID": trace_id,
        "X-Request-ID": request_id,
    }

    async with AsyncCireRagClient(
        base_url=args.rag_url,
        api_key=os.getenv("RAG_SERVICE_SECRET") or os.getenv("RAG_API_KEY"),
        default_headers=default_headers,
        tenant_context=TenantContext(tenant_id=tenant_id),
    ) as client:
        if operation == "resume":
            batch_id = str(args.resume_batch or "").strip()
            if not batch_id:
                if args.non_interactive:
                    raise RuntimeError("--resume-batch is required in non-interactive mode")
                batch_id = _prompt("📝 Batch ID a monitorear: ")
            if not batch_id:
                raise RuntimeError("Batch ID requerido")
            await run_batch_monitoring(client, batch_id, poll_seconds=args.job_poll_seconds)
            return

        if operation == "replay":
            doc_id = str(args.replay_enrichment_doc or "").strip()
            if not doc_id:
                if args.non_interactive:
                    raise RuntimeError(
                        "--replay-enrichment-doc is required in non-interactive mode"
                    )
                doc_id = await _pick_document_for_replay(
                    args=args,
                    client=client,
                    tenant_id=tenant_id,
                    access_token=access_token,
                )
            if not doc_id:
                raise RuntimeError("Document ID requerido para replay")

            res = await client.replay_enrichment(
                doc_id,
                include_visual=not args.replay_no_visual,
                include_graph=not args.replay_no_graph,
                include_raptor=not args.replay_no_raptor,
                tenant_id=tenant_id,
            )

            job_id = str(res.get("job_id") or "").strip() if isinstance(res, dict) else ""
            print(f"✅ Replay encolado para doc={doc_id} | job_id={job_id or 'n/a'}")

            if not args.no_wait and job_id:
                await run_job_monitoring(client, job_id, poll_seconds=args.job_poll_seconds)
            return

        # Regular ingestion
        collection_id = await resolve_collection(args, tenant_id, access_token)

        files_to_upload = []
        if args.file:
            files_to_upload.extend(args.file)
        if args.glob:
            for pattern in args.glob:
                files_to_upload.extend(glob.glob(pattern))

        if not files_to_upload:
            if args.non_interactive:
                print("❌ No hay archivos para subir.")
                return
            path = _prompt("📝 Ruta al archivo o carpeta: ")
            if os.path.isdir(path):
                files_to_upload.extend(glob.glob(os.path.join(path, "*")))
            elif os.path.isfile(path):
                files_to_upload.append(path)
            else:
                print("⚠️ Ruta inválida.")
                return

        print(f"📦 Creando batch para {len(files_to_upload)} archivos...")
        batch = await client.create_ingestion_batch(
            tenant_id=tenant_id, collection_id=collection_id, embedding_mode=args.embedding_mode
        )
        batch_id = batch["id"]
        print(f"✅ Batch creado: {batch_id}")

        for f in files_to_upload:
            print(f"📤 Subiendo {os.path.basename(f)}...")
            await client.upload_file_to_batch(batch_id, f)

        print("🔗 Sellando batch...")
        await client.seal_ingestion_batch(batch_id)
        print("✅ Batch sellado.")

        if not args.no_wait:
            await run_batch_monitoring(client, batch_id, poll_seconds=args.job_poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
