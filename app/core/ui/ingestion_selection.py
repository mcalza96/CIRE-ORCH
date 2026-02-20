from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

from app.core.orch_discovery_client import (
    create_dev_tenant,
    list_authorized_collections,
    list_authorized_tenants,
)
from sdk.python.cire_rag_sdk.client import AsyncCireRagClient


def prompt(message: str) -> str:
    return input(message).strip()


def resolve_operation(args: argparse.Namespace) -> str:
    if args.resume_batch:
        return "resume"
    if args.replay_enrichment_doc:
        return "replay"
    if args.file or args.glob:
        return "ingest"
    return "menu"


def choose_menu_option() -> str:
    print("\n🧭 ¿Qué quieres hacer?")
    print("  1) Nueva ingesta")
    print("  2) Monitorear batch existente")
    print("  3) Replay de enrichment (sin re-ingesta)")
    while True:
        choice = prompt("📝 Elige opción [1-3]: ")
        if choice == "1":
            return "ingest"
        if choice == "2":
            return "resume"
        if choice == "3":
            return "replay"
        print("Opción inválida.")


def resolve_input_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    candidates = [
        value,
        os.path.expanduser(value),
        os.path.abspath(value),
        os.path.abspath(os.path.join(os.getcwd(), "..", value)),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            return candidate
    return ""


def expand_files_from_path(raw_path: str) -> list[str]:
    resolved = resolve_input_path(raw_path)
    if not resolved:
        return []
    if os.path.isfile(resolved):
        return [resolved]
    if os.path.isdir(resolved):
        return [
            candidate
            for candidate in glob.glob(os.path.join(resolved, "*"))
            if os.path.isfile(candidate)
        ]
    return []


def collect_files_interactively() -> list[str]:
    while True:
        path_raw = prompt("📝 Ruta al archivo o carpeta (admite glob, ej: docs/*.pdf): ")
        if not path_raw:
            print("⚠️ Debes indicar una ruta o patrón glob.")
            continue
        if any(token in path_raw for token in ("*", "?", "[")):
            matches = [candidate for candidate in glob.glob(path_raw) if os.path.isfile(candidate)]
            if matches:
                return matches
            print("⚠️ El patrón glob no devolvió archivos válidos.")
            continue
        files = expand_files_from_path(path_raw)
        if files:
            return files
        print("⚠️ Ruta inválida. Usa ruta absoluta, ~/..., relativa o un patrón glob.")


async def resolve_tenant(args: argparse.Namespace, access_token: str) -> str:
    if args.tenant_id:
        return args.tenant_id

    tenants, tenant_discovery_error = await _discover_tenants(
        orchestrator_url=args.orchestrator_url,
        access_token=access_token,
    )
    if tenant_discovery_error is not None:
        print(f"⚠️ Discovery falló: {tenant_discovery_error}")

    if not tenants:
        if args.non_interactive:
            raise RuntimeError("Tenant selection required")
        return await _prompt_tenant_without_discovery(args, access_token)

    if len(tenants) == 1:
        print(f"🏢 Tenant auto-seleccionado: {tenants[0].name}")
        return tenants[0].id
    return await _prompt_tenant_from_list(tenants, args, access_token)


async def resolve_collection(args: argparse.Namespace, tenant_id: str, access_token: str) -> str:
    if args.collection_id:
        return args.collection_id

    while True:
        cols, collections_error = await _discover_collections(
            orchestrator_url=args.orchestrator_url,
            access_token=access_token,
            tenant_id=tenant_id,
        )
        if collections_error is not None:
            if args.non_interactive:
                raise RuntimeError(f"No se pudieron listar colecciones: {collections_error}")
            action = _prompt_collection_discovery_error(collections_error)
            if action == "retry":
                continue
            if action == "create":
                return _prompt_new_collection_name()
            if action == "manual":
                return _prompt_collection_key_manual()
            continue

        if not cols:
            if args.non_interactive:
                return "default"
            action = _prompt_empty_collections_menu()
            if action == "create":
                return _prompt_new_collection_name()
            if action == "manual":
                return _prompt_collection_key_manual()
            continue

        selection = _prompt_collection_from_list(cols)
        if selection == "retry":
            continue
        if selection == "create":
            return _prompt_new_collection_name()
        return selection


def extract_documents(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


def doc_label(doc: dict[str, Any]) -> str:
    filename = str(doc.get("filename") or "").strip()
    if filename:
        return filename
    source_path = str(doc.get("storage_path") or "").strip()
    if source_path:
        return Path(source_path).name
    return str(doc.get("id") or "(sin-id)")


async def pick_document_for_replay(
    *,
    args: argparse.Namespace,
    client: AsyncCireRagClient,
    tenant_id: str,
    access_token: str,
) -> str:
    docs_payload = await client.list_documents(limit=max(1, int(args.documents_limit)))
    all_docs = extract_documents(docs_payload)
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
            raw = prompt(f"📝 Elige colección [1-{len(collections)}]: ")
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
        print(f"  {idx}) {doc_label(doc)} [{status}] ({doc_id}){created_suffix}")

    while True:
        raw = prompt(f"📝 Elige documento [1-{len(selected_docs)}] o pega doc_id: ")
        if raw.isdigit() and 1 <= int(raw) <= len(selected_docs):
            chosen = str(selected_docs[int(raw) - 1].get("id") or "").strip()
            if chosen:
                return chosen
        by_id = str(raw).strip()
        if by_id:
            return by_id
        print("Opción inválida.")


async def _discover_tenants(
    *, orchestrator_url: str, access_token: str
) -> tuple[list[Any], Exception | None]:
    try:
        return await list_authorized_tenants(orchestrator_url, access_token), None
    except Exception as exc:
        return [], exc


async def _discover_collections(
    *,
    orchestrator_url: str,
    access_token: str,
    tenant_id: str,
) -> tuple[list[Any], Exception | None]:
    try:
        return await list_authorized_collections(orchestrator_url, access_token, tenant_id), None
    except Exception as exc:
        return [], exc


def _prompt_non_empty(message: str, missing_message: str) -> str:
    while True:
        value = prompt(message)
        if value:
            return value
        print(missing_message)


def _prompt_new_collection_name() -> str:
    return _prompt_non_empty("📁 Nombre de la nueva colección: ", "Nombre requerido.")


def _prompt_collection_key_manual() -> str:
    return _prompt_non_empty("📁 Collection Key/ID: ", "Collection Key/ID requerido.")


async def _create_dev_tenant_from_prompt(args: argparse.Namespace, access_token: str) -> str:
    while True:
        name = prompt("🏢 Nombre del nuevo tenant (dev): ")
        if not name:
            print("Nombre requerido.")
            continue
        try:
            created = await create_dev_tenant(args.orchestrator_url, access_token, name=name)
        except Exception as exc:
            print(f"⚠️ No se pudo crear tenant dev: {exc}")
            continue
        print(f"✅ Tenant creado: {created.name} ({created.id})")
        return created.id


async def _prompt_tenant_without_discovery(args: argparse.Namespace, access_token: str) -> str:
    print("🏢 No hay tenants autorizados visibles.")
    print("  0) Ingresar tenant manual")
    print("  9) Crear tenant (dev)")
    while True:
        pick = prompt("📝 Elige opción [0/9]: ")
        if pick == "0":
            return _prompt_non_empty("🏢 Tenant ID (UUID): ", "Tenant ID requerido.")
        if pick == "9":
            return await _create_dev_tenant_from_prompt(args, access_token)
        print("Opción inválida.")


async def _prompt_tenant_from_list(
    tenants: list[Any],
    args: argparse.Namespace,
    access_token: str,
) -> str:
    print("🏢 Tenants disponibles:")
    for idx, tenant in enumerate(tenants, start=1):
        print(f"  {idx}) {tenant.name} ({tenant.id})")
    print("  0) Ingresar tenant manual")
    print("  9) Crear tenant (dev)")

    while True:
        opt = prompt(f"📝 Elige tenant [0-{len(tenants)}] o 9: ")
        if opt == "0":
            return _prompt_non_empty("🏢 Tenant ID (UUID): ", "Tenant ID requerido.")
        if opt == "9":
            return await _create_dev_tenant_from_prompt(args, access_token)
        if opt.isdigit() and 1 <= int(opt) <= len(tenants):
            return tenants[int(opt) - 1].id
        print("Opción inválida.")


def _prompt_collection_discovery_error(exc: Exception) -> str:
    print(f"⚠️ No se pudieron listar colecciones: {exc}")
    print("  1) Reintentar")
    print("  2) Crear nueva colección")
    print("  3) Ingresar collection key manual")
    while True:
        retry_opt = prompt("📝 Elige opción [1-3]: ")
        if retry_opt == "1":
            return "retry"
        if retry_opt == "2":
            return "create"
        if retry_opt == "3":
            return "manual"
        print("Opción inválida.")


def _prompt_empty_collections_menu() -> str:
    print("📁 No hay colecciones en este tenant.")
    print("  0) Crear nueva")
    print("  9) Ingresar collection key manual")
    while True:
        opt_empty = prompt("📝 Elige opción [0/9]: ")
        if opt_empty == "0":
            return "create"
        if opt_empty == "9":
            return "manual"
        print("Opción inválida.")


def _prompt_collection_from_list(cols: list[Any]) -> str:
    print("📁 Colecciones disponibles:")
    for idx, collection in enumerate(cols, start=1):
        print(f"  {idx}) {collection.name} ({collection.collection_key or collection.id})")
    print("  0) Crear nueva")
    print("  9) Reintentar listado")

    while True:
        opt = prompt(f"📝 Elige colección [0-{len(cols)}] o 9: ")
        if opt == "0":
            return "create"
        if opt == "9":
            return "retry"
        if opt.isdigit() and 1 <= int(opt) <= len(cols):
            col = cols[int(opt) - 1]
            return col.collection_key or col.id
        print("Opción inválida.")
