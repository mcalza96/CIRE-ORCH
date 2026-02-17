#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate markdown dashboard from benchmark reports"
    )
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).with_name("reports"))
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("reports") / "iso_auditor_dashboard.md"
    )
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    report_paths = sorted(
        args.reports_dir.glob("iso_auditor_benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not report_paths:
        raise SystemExit("No benchmark reports found")

    selected = report_paths[: max(1, int(args.limit))]
    rows: list[dict[str, Any]] = []
    for path in selected:
        payload = _load_report(path)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        rows.append(
            {
                "path": str(path),
                "variant": str(payload.get("variant_label") or ""),
                "generated_at": str(payload.get("generated_at") or ""),
                "citation": _safe_float(metrics.get("citation_marker_rate")),
                "coverage": _safe_float(metrics.get("standard_coverage_rate")),
                "literal": _safe_float(metrics.get("literal_mode_retention_rate")),
                "answerable": _safe_float(metrics.get("answerable_rate")),
                "false_positive": _safe_float(metrics.get("false_positive_partial_rate")),
                "latency": _safe_float(metrics.get("latency_p95_ms")),
                "checks_ok": bool(checks) and all(bool(v) for v in checks.values()),
            }
        )

    latest = rows[0]
    best_cov = max(rows, key=lambda row: row["coverage"])
    best_lit = max(rows, key=lambda row: row["literal"])

    lines: list[str] = []
    lines.append("# ISO Auditor Dashboard")
    lines.append("")
    lines.append(f"- Latest report: `{latest['path']}`")
    lines.append(f"- Best coverage: `{best_cov['variant']}` ({best_cov['coverage']:.2f})")
    lines.append(f"- Best literal retention: `{best_lit['variant']}` ({best_lit['literal']:.2f})")
    lines.append("")
    lines.append("## Latest Metrics")
    lines.append("")
    lines.append(
        f"- citation={latest['citation']:.2f} | coverage={latest['coverage']:.2f} | literal={latest['literal']:.2f} | "
        f"answerable={latest['answerable']:.2f} | false_positive={latest['false_positive']:.2f} | p95_ms={latest['latency']:.2f}"
    )
    lines.append("")
    lines.append("## Recent Runs")
    lines.append("")
    lines.append(
        "| variant | generated_at | citation | coverage | literal | answerable | false_positive | p95_ms | checks |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['generated_at']} | {row['citation']:.2f} | {row['coverage']:.2f} | {row['literal']:.2f} | "
            f"{row['answerable']:.2f} | {row['false_positive']:.2f} | {row['latency']:.2f} | {'OK' if row['checks_ok'] else 'FAIL'} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
