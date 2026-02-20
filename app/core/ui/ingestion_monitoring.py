from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from app.core.observability.ingestion_utils import human_ingestion_stage
from app.core.ui.ingestion_models import (
    BATCH_TERMINAL_STATES,
    HEARTBEAT_SECONDS,
    JOB_TERMINAL_STATES,
    PollStep,
    _BATCH_STAGE_LABELS,
    _JOB_STAGE_LABELS,
)
from sdk.python.cire_rag_sdk.client import AsyncCireRagClient


def _status_timestamp(status: dict[str, Any]) -> str:
    updated = str(status.get("updated_at") or "").strip()
    if updated:
        return updated
    return str(status.get("created_at") or "").strip()


def _detect_resilient_mode(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key in (
            "fallback",
            "fallback_used",
            "fallback_reason",
            "fallback_blocked_by_literal_lock",
        ):
            if payload.get(key):
                return True

        for key in ("error_codes", "warning_codes", "warnings"):
            value = payload.get(key)
            if isinstance(value, list) and "fallback" in " ".join(str(x).lower() for x in value):
                return True

        return any(_detect_resilient_mode(value) for value in payload.values())

    if isinstance(payload, list):
        return any(_detect_resilient_mode(item) for item in payload)

    if isinstance(payload, str):
        return "fallback" in payload.lower()
    return False


async def generic_poller(
    *,
    title: str,
    poll_seconds: float,
    fetch_status_fn: Callable[[], Awaitable[Any]],
    process_status_fn: Callable[[Any, float], PollStep],
    error_prefix: str,
) -> None:
    print(title)
    next_poll_at = 0.0
    last_event_at = time.monotonic()
    last_heartbeat_at = 0.0
    started_at = time.monotonic()
    heartbeat_label = "Procesando"
    resilient_mode_reported = False

    while True:
        now = time.monotonic()
        if now >= next_poll_at:
            try:
                status = await fetch_status_fn()
                if not resilient_mode_reported and _detect_resilient_mode(status):
                    print(
                        "🛟 Modo resiliente activado: se aplicaron rutas de fallback para completar."
                    )
                    resilient_mode_reported = True

                step = process_status_fn(status, started_at)
                heartbeat_label = step.heartbeat_label or heartbeat_label
                if step.event_emitted:
                    last_event_at = time.monotonic()
                if step.terminal:
                    print(step.terminal_message)
                    break
            except Exception as exc:
                print(f"⚠️ {error_prefix}: {exc}")
                last_event_at = time.monotonic()

            next_poll_at = time.monotonic() + max(1.0, float(poll_seconds))

        idle = time.monotonic() - last_event_at
        if (
            idle >= HEARTBEAT_SECONDS
            and (time.monotonic() - last_heartbeat_at) >= HEARTBEAT_SECONDS
        ):
            elapsed = round(time.monotonic() - started_at, 1)
            print(f"⏳ {heartbeat_label}... {elapsed}s")
            last_heartbeat_at = time.monotonic()

        await asyncio.sleep(0.35)


async def run_batch_monitoring(
    client: AsyncCireRagClient,
    batch_id: str,
    *,
    poll_seconds: float,
) -> None:
    last_snapshot = ""
    last_stage = ""

    def _process_batch_status(status: Any, started_at: float) -> PollStep:
        nonlocal last_snapshot, last_stage
        if not isinstance(status, dict):
            return PollStep(heartbeat_label="Procesando pipeline")

        batch = status.get("batch", {})
        obs = status.get("observability", {})
        state = batch.get("status", "unknown")
        percent = obs.get("progress_percent", 0.0)
        stage = str(obs.get("dominant_stage", "OTHER") or "OTHER").upper()
        heartbeat_label = _BATCH_STAGE_LABELS.get(stage, "Procesando pipeline")
        event_emitted = False

        if stage and stage != last_stage:
            if last_stage:
                print(
                    f"✅ {(_BATCH_STAGE_LABELS.get(last_stage) or human_ingestion_stage(last_stage))} completado"
                )
            print(f"🧠 {(_BATCH_STAGE_LABELS.get(stage) or human_ingestion_stage(stage))}...")
            last_stage = stage
            event_emitted = True

        snapshot = f"{state}|{percent}|{stage}"
        if snapshot != last_snapshot:
            print(
                f"📡 Estado: {state} | Progreso: {percent}% | Etapa: {human_ingestion_stage(stage)}"
            )
            last_snapshot = snapshot
            event_emitted = True

        if str(state or "").strip().lower() in BATCH_TERMINAL_STATES:
            elapsed = round(time.monotonic() - started_at, 1)
            return PollStep(
                heartbeat_label=heartbeat_label,
                event_emitted=True,
                terminal=True,
                terminal_message=f"✅ Batch finalizado con estado: {state} ({elapsed}s)",
            )

        return PollStep(heartbeat_label=heartbeat_label, event_emitted=event_emitted)

    await generic_poller(
        title=f"🔎 Monitoreando batch {batch_id}...",
        poll_seconds=poll_seconds,
        fetch_status_fn=lambda: client.get_batch_status(batch_id),
        process_status_fn=_process_batch_status,
        error_prefix="Error monitoreando",
    )


async def run_job_monitoring(
    client: AsyncCireRagClient,
    job_id: str,
    *,
    poll_seconds: float,
) -> None:
    last_snapshot = ""
    last_state = ""

    def _process_job_status(status: Any, started_at: float) -> PollStep:
        nonlocal last_snapshot, last_state
        if not isinstance(status, dict):
            return PollStep(heartbeat_label="Procesando enrichment")

        state = str(status.get("status") or "unknown").strip().lower()
        job_type = str(status.get("job_type") or "unknown").strip()
        err = str(status.get("error_message") or "").strip()
        stamp = _status_timestamp(status)
        heartbeat_label = _JOB_STAGE_LABELS.get(state, "Procesando enrichment")
        event_emitted = False

        if state != last_state and state in _JOB_STAGE_LABELS:
            print(f"🧠 {_JOB_STAGE_LABELS[state]}...")
            last_state = state
            event_emitted = True

        snapshot = f"{job_type}|{state}|{err}|{stamp}"
        if snapshot != last_snapshot:
            suffix = f" | error={err}" if err else ""
            stamp_suffix = f" | updated_at={stamp}" if stamp else ""
            print(f"📡 Job: {job_type} | Estado: {state}{stamp_suffix}{suffix}")
            last_snapshot = snapshot
            event_emitted = True

        if state in JOB_TERMINAL_STATES:
            elapsed = round(time.monotonic() - started_at, 1)
            terminal_message = (
                f"✅ Replay finalizado correctamente ({elapsed}s)."
                if state == "completed"
                else f"❌ Replay terminó con estado: {state} ({elapsed}s)"
            )
            return PollStep(
                heartbeat_label=heartbeat_label,
                event_emitted=True,
                terminal=True,
                terminal_message=terminal_message,
            )

        return PollStep(heartbeat_label=heartbeat_label, event_emitted=event_emitted)

    await generic_poller(
        title=f"🔎 Monitoreando job {job_id}...",
        poll_seconds=poll_seconds,
        fetch_status_fn=lambda: client.get_ingestion_job_status(job_id),
        process_status_fn=_process_job_status,
        error_prefix="Error monitoreando job",
    )
