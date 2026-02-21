from __future__ import annotations
import httpx
import json
import argparse
from typing import Any

def human_ingestion_stage(stage: str) -> str:
    normalized = str(stage or "").strip().upper()
    mapping = {
        "INGEST": "Preparando",
        "PERSIST": "Guardando",
        "VISUAL": "Enriquecimiento visual",
        "RAPTOR": "Resumen jerárquico",
        "GRAPH": "Extracción de grafo",
        "DONE": "Listo",
        "ERROR": "Atención requerida",
        "OTHER": "Analizando estructura",
        "QUEUED": "En cola",
    }
    return mapping.get(normalized, str(stage or "Analizando"))

def human_batch_mode(seal_state: str) -> str:
    mapping = {
        "omitido (overwritable)": "Sobrescritura habilitada",
        "omitido (resume-batch)": "Reanudación de batch",
    }
    return mapping.get(seal_state, seal_state)

def _obs_headers(access_token: str | None, tenant_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers

async def show_ingestion_overview(
    *,
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_id: str,
    access_token: str,
) -> None:
    base = orchestrator_url.rstrip("/")
    headers = _obs_headers(access_token, tenant_id=tenant_id)
    try:
        active_resp = await client.get(
            f"{base}/api/v1/observability/batches/active",
            params={"tenant_id": tenant_id, "limit": 5},
            headers=headers,
        )
        active_resp.raise_for_status()
        active_payload = active_resp.json()
    except Exception as exc:
        print(f"❌ No se pudo obtener batches activos: {exc}")
        return

    items = active_payload.get("items") if isinstance(active_payload, dict) else []
    if not isinstance(items, list):
        items = []

    print("📡 Ingestion overview")
    print(f"   tenant={tenant_id}")
    print(f"   active_batches={len(items)}")
    print("   [Batches]")
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        batch = item.get("batch") if isinstance(item.get("batch"), dict) else {}
        obs = item.get("observability") if isinstance(item.get("observability"), dict) else {}
        batch_id = str(batch.get("id") or batch.get("batch_id") or "unknown")
        status = str(batch.get("status") or "unknown")
        percent = float(obs.get("progress_percent") or 0.0)
        stage = str(obs.get("dominant_stage") or "OTHER")
        stage_human = human_ingestion_stage(stage)
        eta = int(obs.get("eta_seconds") or 0)
        print(
            f"   - {batch_id}: status={status} progress={percent}% stage={stage_human} eta={eta}s"
        )

async def watch_batch_stream(
    *,
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_id: str,
    access_token: str,
    batch_id: str,
) -> None:
    base = orchestrator_url.rstrip("/")
    url = f"{base}/api/v1/observability/batches/{batch_id}/stream"
    headers = _obs_headers(access_token, tenant_id=tenant_id)
    params = {"tenant_id": tenant_id, "interval_ms": 1500}
    print(f"🔎 Watching batch {batch_id} ...")

    current_event = "message"
    try:
        async with client.stream(
            "GET", url, params=params, headers=headers, timeout=None
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                text = await response.aread()
                print(
                    f"❌ watch failed (HTTP {response.status_code}): {text.decode('utf-8', errors='ignore')[:300]}"
                )
                return
            async for raw_line in response.aiter_lines():
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip() or "message"
                    continue
                if not line.startswith("data:"):
                    continue
                payload_text = line.split(":", 1)[1].strip()
                try:
                    payload = json.loads(payload_text) if payload_text else {}
                except Exception:
                    continue
                if current_event == "snapshot":
                    progress = payload.get("progress") if isinstance(payload, dict) else {}
                    batch = (
                        progress.get("batch")
                        if isinstance(progress, dict) and isinstance(progress.get("batch"), dict)
                        else {}
                    )
                    obs = (
                        progress.get("observability")
                        if isinstance(progress, dict)
                        and isinstance(progress.get("observability"), dict)
                        else {}
                    )
                    status = str(batch.get("status") or "unknown")
                    percent = float(obs.get("progress_percent") or 0.0)
                    stage = str(obs.get("dominant_stage") or "OTHER")
                    stage_human = human_ingestion_stage(stage)
                    eta = int(obs.get("eta_seconds") or 0)
                    stalled = bool(obs.get("stalled", False))
                    print(
                        f"📡 status={status} progress={percent}% stage={stage_human} eta={eta}s stalled={stalled}"
                    )
                elif current_event == "done":
                    print("✅ Batch completed!")
                    return
                elif current_event == "error":
                    print(f"❌ Batch error: {payload.get('message', 'Unknown error')}")
                    return
    except Exception as exc:
        print(f"❌ stream watch error: {exc}")
