#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRACKED = (
    "citation_marker_rate",
    "standard_coverage_rate",
    "coverage_strict_rate",
    "coverage_partial_honest_rate",
    "literal_mode_retention_rate",
    "answerable_rate",
    "false_positive_partial_rate",
    "latency_p95_ms",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid report payload: {path}")
    return payload


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark reports")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    before_metrics = before.get("metrics") if isinstance(before.get("metrics"), dict) else {}
    after_metrics = after.get("metrics") if isinstance(after.get("metrics"), dict) else {}

    deltas: dict[str, dict[str, float]] = {}
    for key in TRACKED:
        b = _safe_float(before_metrics.get(key))
        a = _safe_float(after_metrics.get(key))
        deltas[key] = {
            "before": round(b, 2),
            "after": round(a, 2),
            "delta": round(a - b, 2),
        }

    summary = {
        "before": str(args.before),
        "after": str(args.after),
        "deltas": deltas,
        "regressions": [
            key
            for key, item in deltas.items()
            if (key in {"false_positive_partial_rate", "latency_p95_ms"} and item["delta"] > 0)
            or (key not in {"false_positive_partial_rate", "latency_p95_ms"} and item["delta"] < 0)
        ],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
