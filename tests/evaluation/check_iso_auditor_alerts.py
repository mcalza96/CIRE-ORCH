#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_report(path: Path | None, reports_dir: Path) -> Path:
    if path is not None:
        return path
    candidates = sorted(
        reports_dir.glob("iso_auditor_benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No benchmark reports found")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check alerts for ISO benchmark report")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).with_name("reports"))
    parser.add_argument("--min-citation", type=float, default=98.0)
    parser.add_argument("--min-coverage", type=float, default=90.0)
    parser.add_argument("--min-literal", type=float, default=95.0)
    parser.add_argument("--min-answerable", type=float, default=90.0)
    parser.add_argument("--max-false-positive", type=float, default=10.0)
    args = parser.parse_args()

    report_path = _resolve_report(args.report, args.reports_dir)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    citation = _safe_float(metrics.get("citation_marker_rate"))
    coverage = _safe_float(metrics.get("standard_coverage_rate"))
    literal = _safe_float(metrics.get("literal_mode_retention_rate"))
    answerable = _safe_float(metrics.get("answerable_rate"))
    false_positive = _safe_float(metrics.get("false_positive_partial_rate"))

    alerts: list[str] = []
    if citation < args.min_citation:
        alerts.append("citation_marker_rate_below_threshold")
    if coverage < args.min_coverage:
        alerts.append("standard_coverage_rate_below_threshold")
    if literal < args.min_literal:
        alerts.append("literal_mode_retention_rate_below_threshold")
    if answerable < args.min_answerable:
        alerts.append("answerable_rate_below_threshold")
    if false_positive > args.max_false_positive:
        alerts.append("false_positive_partial_rate_above_threshold")

    output = {
        "report": str(report_path),
        "alerts": alerts,
        "metrics": {
            "citation_marker_rate": citation,
            "standard_coverage_rate": coverage,
            "literal_mode_retention_rate": literal,
            "answerable_rate": answerable,
            "false_positive_partial_rate": false_positive,
        },
        "thresholds": {
            "min_citation": args.min_citation,
            "min_coverage": args.min_coverage,
            "min_literal": args.min_literal,
            "min_answerable": args.min_answerable,
            "max_false_positive": args.max_false_positive,
        },
    }
    print(json.dumps(output, ensure_ascii=True))
    return 2 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
