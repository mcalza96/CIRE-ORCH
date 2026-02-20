from __future__ import annotations

from typing import Any

import structlog

from app.agent.error_codes import RETRIEVAL_CODE_LOW_SCORE, merge_error_codes
from app.core.config import settings


logger = structlog.get_logger(__name__)


def _safe_score(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _item_score_space(item: dict[str, Any]) -> str:
    score_space = str(item.get("score_space") or "").strip().lower()
    if score_space:
        return score_space
    metadata_raw = item.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    score_space = str(metadata.get("score_space") or "").strip().lower()
    if score_space:
        return score_space
    row_raw = metadata.get("row")
    row = row_raw if isinstance(row_raw, dict) else {}
    return str(row.get("score_space") or "").strip().lower()


def filter_items_by_min_score(
    items: list[dict[str, Any]],
    *,
    threshold: float | None,
    trace_target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if threshold is None:
        return items

    kept: list[dict[str, Any]] = []
    dropped = 0
    dropped_invalid_score = 0
    scored_dropped: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        score_raw = item.get("score")
        if score_raw is None:
            score_raw = item.get("similarity")
        score_space = _item_score_space(item)
        if score_space in {"rrf", "mixed"}:
            kept.append(item)
            continue
        score = _safe_score(score_raw)
        if score_raw is None or score is None:
            if score_raw is not None and score is None:
                dropped_invalid_score += 1
            kept.append(item)
            continue
        if score >= threshold:
            kept.append(item)
        else:
            dropped += 1
            scored_dropped.append((score, item))

    backstop_applied = False
    backstop_enabled = bool(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_ENABLED", False))
    backstop_top_n = max(1, int(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_TOP_N", 6) or 6))
    if not kept and dropped > 0 and backstop_enabled:
        top_n = backstop_top_n
        scored_dropped.sort(key=lambda pair: pair[0], reverse=True)
        kept = [item for _, item in scored_dropped[:top_n]]
        backstop_applied = bool(kept)

    logger.info(
        "retrieval_min_score_filter",
        threshold=threshold,
        kept=len(kept),
        dropped=dropped,
        dropped_invalid_score=dropped_invalid_score,
        backstop_applied=backstop_applied,
        top_dropped_score=(max((score for score, _ in scored_dropped), default=None)),
    )

    if isinstance(trace_target, dict):
        trace_target["min_score_filter"] = {
            "threshold": threshold,
            "kept": len(kept),
            "dropped": dropped,
            "dropped_invalid_score": dropped_invalid_score,
            "backstop_applied": backstop_applied,
            "backstop_top_n": (backstop_top_n if backstop_applied else 0),
        }
        if dropped > 0 and not kept:
            trace_target["error_codes"] = merge_error_codes(
                trace_target.get("error_codes"),
                [RETRIEVAL_CODE_LOW_SCORE],
            )
    return kept
