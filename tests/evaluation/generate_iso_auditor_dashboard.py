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
        metrics_raw = payload.get("metrics")
        checks_raw = payload.get("checks")
        metrics: dict[str, Any] = metrics_raw if isinstance(metrics_raw, dict) else {}
        checks: dict[str, Any] = checks_raw if isinstance(checks_raw, dict) else {}
        rows.append(
            {
                "path": str(path),
                "variant": str(payload.get("variant_label") or ""),
                "generated_at": str(payload.get("generated_at") or ""),
                "citation": _safe_float(metrics.get("citation_marker_rate")),
                "citation_suff": _safe_float(metrics.get("citation_sufficiency_rate")),
                "coverage": _safe_float(metrics.get("standard_coverage_rate")),
                "semantic": _safe_float(metrics.get("semantic_recall_rate")),
                "clause": _safe_float(metrics.get("clause_recall_rate")),
                "hallucination_guard": _safe_float(metrics.get("hallucination_guard_rate")),
                "literal_obedience": _safe_float(metrics.get("literal_obedience_rate")),
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
        f"- citation={latest['citation']:.2f} | citation_suff={latest['citation_suff']:.2f} | coverage={latest['coverage']:.2f} | semantic={latest['semantic']:.2f} | clause={latest['clause']:.2f}"
    )
    lines.append(
        f"- hallucination_guard={latest['hallucination_guard']:.2f} | literal_obedience={latest['literal_obedience']:.2f} | literal_mode_retention={latest['literal']:.2f} | answerable={latest['answerable']:.2f} | false_positive={latest['false_positive']:.2f} | p95_ms={latest['latency']:.2f}"
    )
    lines.append("")
    lines.append("## Recent Runs")
    lines.append("")
    lines.append(
        "| variant | generated_at | citation | citation_suff | coverage | semantic | clause | hallucination_guard | literal_obedience | literal | answerable | false_positive | p95_ms | checks |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['generated_at']} | {row['citation']:.2f} | {row['citation_suff']:.2f} | {row['coverage']:.2f} | {row['semantic']:.2f} | {row['clause']:.2f} | {row['hallucination_guard']:.2f} | {row['literal_obedience']:.2f} | {row['literal']:.2f} | "
            f"{row['answerable']:.2f} | {row['false_positive']:.2f} | {row['latency']:.2f} | {'OK' if row['checks_ok'] else 'FAIL'} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
