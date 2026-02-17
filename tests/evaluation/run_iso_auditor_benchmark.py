#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


DEFAULT_DATASET = Path(__file__).with_name("iso_auditor_benchmark.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("reports")
CITATION_RE = re.compile(r"\b[CR]\d+\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    expected_mode: str
    mode: str
    final_mode: str
    has_citation_marker: bool
    coverage_ok: bool
    literal_mode_retained: bool
    answerable: bool
    false_positive_partial: bool
    latency_ms: float
    status_code: int
    error: str | None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_latency_ms(payload: dict[str, Any]) -> float:
    retrieval_plan = payload.get("retrieval_plan")
    if isinstance(retrieval_plan, dict):
        timings = retrieval_plan.get("timings_ms")
        if isinstance(timings, dict):
            for key in ("total", "hybrid", "multi_query_primary", "multi_query_fallback"):
                if key in timings:
                    return _safe_float(timings.get(key), 0.0)
    return 0.0


def _mode_retained(category: str, final_mode: str, expected_mode: str) -> bool:
    if category != "literal":
        return True
    final_norm = str(final_mode or "").strip().lower()
    expected_norm = str(expected_mode or "").strip().lower()
    return final_norm.startswith("literal") or final_norm == expected_norm


def _coverage_ok(
    expected_standards: list[str],
    missing_scopes: list[str],
    answer: str,
) -> bool:
    if not expected_standards:
        return True
    missing = {str(item).strip().upper() for item in missing_scopes if str(item).strip()}
    expected = [str(item).strip().upper() for item in expected_standards if str(item).strip()]
    if any(item in missing for item in expected):
        return False

    blob = str(answer or "").upper()
    for scope in expected:
        numeric = "".join(ch for ch in scope if ch.isdigit())
        if scope in blob:
            continue
        if numeric and numeric in blob:
            continue
        return False
    return True


def _evaluate_case(case: dict[str, Any], status_code: int, payload: dict[str, Any]) -> CaseResult:
    category = str(case.get("category") or "").strip().lower()
    expected_mode = str(case.get("expected_mode") or "").strip()
    expected_standards = [
        str(item).strip()
        for item in (case.get("expected_standards") or [])
        if str(item).strip()
    ]
    require_citations = bool(case.get("require_citations", True))
    require_full_coverage = bool(case.get("require_full_coverage", False))

    if status_code != 200:
        return CaseResult(
            case_id=str(case.get("id") or ""),
            category=category,
            expected_mode=expected_mode,
            mode="",
            final_mode="",
            has_citation_marker=False,
            coverage_ok=False,
            literal_mode_retained=False,
            answerable=False,
            false_positive_partial=False,
            latency_ms=0.0,
            status_code=status_code,
            error=f"http_{status_code}",
        )

    answer = str(payload.get("answer") or "")
    mode = str(payload.get("mode") or "").strip()
    retrieval_plan = payload.get("retrieval_plan") if isinstance(payload.get("retrieval_plan"), dict) else {}
    final_mode = str(retrieval_plan.get("final_mode") or mode).strip()
    missing_scopes = (
        retrieval_plan.get("missing_scopes")
        if isinstance(retrieval_plan.get("missing_scopes"), list)
        else []
    )
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    context_chunks = (
        payload.get("context_chunks") if isinstance(payload.get("context_chunks"), list) else []
    )
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    accepted = bool(validation.get("accepted", True))

    has_citation_marker = bool(citations) or bool(CITATION_RE.search(answer))
    if not require_citations:
        has_citation_marker = True

    coverage_ok = _coverage_ok(expected_standards, list(missing_scopes), answer)
    literal_mode_retained = _mode_retained(category, final_mode, expected_mode)
    answerable = bool(accepted and context_chunks and has_citation_marker)
    false_positive_partial = bool(
        require_full_coverage
        and accepted
        and (not coverage_ok or not has_citation_marker)
    )

    return CaseResult(
        case_id=str(case.get("id") or ""),
        category=category,
        expected_mode=expected_mode,
        mode=mode,
        final_mode=final_mode,
        has_citation_marker=has_citation_marker,
        coverage_ok=coverage_ok,
        literal_mode_retained=literal_mode_retained,
        answerable=answerable,
        false_positive_partial=false_positive_partial,
        latency_ms=_extract_latency_ms(payload),
        status_code=status_code,
        error=None,
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _p95(values: list[float]) -> float:
    positives = [float(v) for v in values if float(v) > 0.0]
    if not positives:
        return 0.0
    if len(positives) == 1:
        return positives[0]
    return round(float(quantiles(positives, n=100, method="inclusive")[94]), 2)


async def _run_case(
    client: "httpx.AsyncClient",
    *,
    base_url: str,
    tenant_id: str,
    collection_id: str | None,
    case: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    payload = {
        "query": str(case.get("query") or "").strip(),
        "tenant_id": tenant_id,
        "collection_id": collection_id,
    }
    response = await client.post(f"{base_url.rstrip('/')}/api/v1/knowledge/answer", json=payload)
    body: dict[str, Any] = {}
    try:
        raw = response.json()
        body = raw if isinstance(raw, dict) else {"raw": raw}
    except Exception:
        body = {"raw_text": response.text[:2000]}
    return response.status_code, body


async def main() -> int:
    parser = argparse.ArgumentParser(description="Runs iso_auditor benchmark against ORCH API.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=os.getenv("ORCH_BASE_URL", "http://localhost:8001"))
    parser.add_argument("--tenant-id", default=os.getenv("ORCH_BENCH_TENANT_ID", ""))
    parser.add_argument("--collection-id", default=os.getenv("ORCH_BENCH_COLLECTION_ID"))
    parser.add_argument("--access-token", default=os.getenv("AUTH_BEARER_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--variant-label", default=os.getenv("ORCH_BENCH_VARIANT", "baseline"))
    parser.add_argument("--fail-on-thresholds", action="store_true")
    args = parser.parse_args()

    try:
        import httpx
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Missing dependency 'httpx'. Run with project venv (e.g. ./venv/bin/python)."
        ) from exc

    if not args.tenant_id.strip():
        raise SystemExit("Missing --tenant-id (or ORCH_BENCH_TENANT_ID).")

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases_raw = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(cases_raw, list) or not cases_raw:
        raise SystemExit(f"Dataset without cases: {args.dataset}")

    cases = [item for item in cases_raw if isinstance(item, dict)]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    headers: dict[str, str] = {}
    token = str(args.access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = httpx.Timeout(args.timeout_seconds, connect=min(10.0, args.timeout_seconds))
    results: list[CaseResult] = []
    raw_rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for case in cases:
            status_code, payload = await _run_case(
                client,
                base_url=args.base_url,
                tenant_id=args.tenant_id.strip(),
                collection_id=(str(args.collection_id).strip() if args.collection_id else None),
                case=case,
            )
            evaluated = _evaluate_case(case, status_code, payload)
            results.append(evaluated)
            raw_rows.append(
                {
                    "id": evaluated.case_id,
                    "category": evaluated.category,
                    "status_code": evaluated.status_code,
                    "mode": evaluated.mode,
                    "final_mode": evaluated.final_mode,
                    "has_citation_marker": evaluated.has_citation_marker,
                    "coverage_ok": evaluated.coverage_ok,
                    "literal_mode_retained": evaluated.literal_mode_retained,
                    "answerable": evaluated.answerable,
                    "false_positive_partial": evaluated.false_positive_partial,
                    "latency_ms": evaluated.latency_ms,
                    "error": evaluated.error,
                }
            )

    total = len(results)
    literal_rows = [r for r in results if r.category == "literal"]
    coverage_rows = [r for r in results if r.expected_mode and r.coverage_ok is not None]
    require_full = [
        c for c in cases if bool(c.get("require_full_coverage", False))
    ]
    require_full_ids = {str(item.get("id") or "") for item in require_full}
    require_full_results = [r for r in results if r.case_id in require_full_ids]

    citation_marker_rate = _rate(sum(r.has_citation_marker for r in results), total)
    standard_coverage_rate = _rate(sum(r.coverage_ok for r in coverage_rows), len(coverage_rows))
    literal_mode_retention_rate = _rate(
        sum(r.literal_mode_retained for r in literal_rows), len(literal_rows)
    )
    answerable_rate = _rate(sum(r.answerable for r in results), total)
    false_positive_partial_rate = _rate(
        sum(r.false_positive_partial for r in require_full_results),
        len(require_full_results),
    )

    thresholds = {
        "citation_marker_rate": 98.0,
        "standard_coverage_rate": 90.0,
        "literal_mode_retention_rate": 95.0,
    }
    checks = {
        "citation_marker_rate": citation_marker_rate >= thresholds["citation_marker_rate"],
        "standard_coverage_rate": standard_coverage_rate >= thresholds["standard_coverage_rate"],
        "literal_mode_retention_rate": literal_mode_retention_rate
        >= thresholds["literal_mode_retention_rate"],
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant_label": args.variant_label,
        "dataset": str(args.dataset),
        "base_url": args.base_url,
        "tenant_id": args.tenant_id,
        "collection_id": args.collection_id,
        "total_cases": total,
        "metrics": {
            "citation_marker_rate": citation_marker_rate,
            "standard_coverage_rate": standard_coverage_rate,
            "literal_mode_retention_rate": literal_mode_retention_rate,
            "answerable_rate": answerable_rate,
            "false_positive_partial_rate": false_positive_partial_rate,
            "latency_p95_ms": _p95([row.latency_ms for row in results]),
        },
        "thresholds": thresholds,
        "checks": checks,
        "failed_cases": [row for row in raw_rows if row["status_code"] != 200],
        "cases": raw_rows,
    }

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = DEFAULT_REPORTS_DIR / f"iso_auditor_benchmark_{args.variant_label}_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(out_path), "metrics": report["metrics"], "checks": checks}, ensure_ascii=True))

    if args.fail_on_thresholds and not all(checks.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
