#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CLAUSE_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
STD_RE = re.compile(r"\bISO\s*[-:]?\s*(9001|14001|45001)\b", flags=re.IGNORECASE)
DEFAULT_DATASET = Path(__file__).with_name("iso_auditor_benchmark.json")


def _extract_targets(query: str) -> tuple[list[str], list[str]]:
    clauses = sorted(set(CLAUSE_RE.findall(query or "")))
    standards = sorted({f"ISO {m}" for m in STD_RE.findall(query or "")})
    return standards, clauses


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_dataset_cases(dataset_path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("cases") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id") or "").strip()
        if not cid:
            continue
        out[cid] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build directed reingest targets from benchmark report"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else []
    if not isinstance(cases, list):
        cases = []
    dataset_cases = _load_dataset_cases(args.dataset)

    targets: dict[str, dict[str, Any]] = {}
    unresolved_cases: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = str(case.get("id") or "").strip()
        if not cid:
            continue
        status_code = _safe_int(case.get("status_code"), default=0)
        coverage_ok = bool(case.get("coverage_ok", False))
        validation_accepted = bool(case.get("validation_accepted", True))
        has_citation_marker = bool(case.get("has_citation_marker", True))

        reasons: list[str] = []
        if status_code != 200:
            reasons.append("http_error")
        if not coverage_ok:
            reasons.append("coverage_gap")
        if not validation_accepted:
            reasons.append("validation_rejected")
        if not has_citation_marker:
            reasons.append("missing_citation_marker")
        if not reasons:
            continue

        report_query = str(case.get("query") or "").strip()
        dataset_case = dataset_cases.get(cid) if dataset_cases else None
        dataset_query = (
            str(dataset_case.get("query") or "").strip() if isinstance(dataset_case, dict) else ""
        )
        query = report_query or dataset_query

        standards, clauses = _extract_targets(query)
        if not standards:
            expected_standards = case.get("expected_standards")
            if isinstance(expected_standards, list):
                standards = [
                    str(item).strip().upper().replace("ISO-", "ISO ")
                    for item in expected_standards
                    if str(item).strip()
                ]
        if not standards:
            unresolved_cases.append(cid)

        targets[cid] = {
            "query": query,
            "standards": standards,
            "clauses": clauses,
            "reasons": reasons,
            "status_code": status_code,
            "coverage_ok": coverage_ok,
            "validation_accepted": validation_accepted,
            "has_citation_marker": has_citation_marker,
            "validation_issues": case.get("validation_issues", []),
        }

    grouped: dict[str, dict[str, Any]] = {}
    for item in targets.values():
        for std in item.get("standards", []):
            bucket = grouped.setdefault(
                std,
                {
                    "clauses": set(),
                    "queries": 0,
                    "missing_citation_marker": 0,
                    "validation_rejected": 0,
                    "coverage_gap": 0,
                    "http_error": 0,
                },
            )
            bucket["queries"] += 1
            for clause in item.get("clauses", []):
                bucket["clauses"].add(clause)
            for reason in item.get("reasons", []):
                if reason in bucket:
                    bucket[reason] += 1

    standard_priority = []
    for std, info in grouped.items():
        score = (
            int(info.get("queries", 0)) * 4
            + int(info.get("coverage_gap", 0)) * 2
            + int(info.get("validation_rejected", 0)) * 3
            + int(info.get("missing_citation_marker", 0))
            + int(info.get("http_error", 0)) * 5
        )
        standard_priority.append({"standard": std, "priority_score": score})
    standard_priority.sort(key=lambda item: item["priority_score"], reverse=True)

    output = {
        "source_report": str(args.report),
        "source_dataset": str(args.dataset),
        "target_cases": targets,
        "by_standard": {
            std: {
                "queries": info["queries"],
                "clauses": sorted(info["clauses"]),
                "coverage_gap": int(info.get("coverage_gap", 0)),
                "validation_rejected": int(info.get("validation_rejected", 0)),
                "missing_citation_marker": int(info.get("missing_citation_marker", 0)),
                "http_error": int(info.get("http_error", 0)),
            }
            for std, info in grouped.items()
        },
        "priority": standard_priority,
        "unresolved_cases": unresolved_cases,
        "notes": [
            "Use this file to drive directed reingestion of missing clauses per standard.",
            "Prioritize standards by priority score, then missing clauses.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
