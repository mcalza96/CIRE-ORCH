from __future__ import annotations

from typing import Any


def compact_error(value: Any, *, limit: int = 320) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def emit_event(logger: Any, event: str, *, level: str = "info", **fields: Any) -> None:
    event_name = str(event)
    payload: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if key == "event":
            continue
        payload[key] = value

    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        try:
            log_fn(event_name, **payload)
            return
        except TypeError:
            try:
                log_fn(event_name, extra={"event": event_name, **payload})
                return
            except TypeError:
                log_fn(event_name)
                return
    logger.info(event_name)
