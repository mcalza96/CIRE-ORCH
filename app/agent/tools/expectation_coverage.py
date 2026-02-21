from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agent.components.parsing import extract_row_standard
from app.agent.models import RetrievalPlan, ToolResult
from app.agent.tools.base import ToolRuntimeContext


def _clause_ref_matches(requested: str, candidate: str) -> bool:
    req = str(requested or "").strip()
    cand = str(candidate or "").strip()
    if not req or not cand:
        return False
    return cand == req or cand.startswith(f"{req}.")


def _row_clause_refs(row: dict[str, Any]) -> set[str]:
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    refs: set[str] = set()
    for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
        value = str(meta.get(key) or "").strip()
        if value:
            refs.add(value)
    clause_refs = meta.get("clause_refs")
    if isinstance(clause_refs, list):
        for item in clause_refs:
            value = str(item or "").strip()
            if value:
                refs.add(value)
    return refs


def _standard_allowed(row_standard: str, expected_scopes: list[str]) -> bool:
    if not expected_scopes:
        return True
    row_std = str(row_standard or "").upper().strip()
    if not row_std:
        return False
    return any(scope.upper() in row_std for scope in expected_scopes)


def _marker_hits(blob: str, markers: list[str]) -> list[str]:
    hits: list[str] = []
    lowered = blob.lower()
    for marker in markers:
        value = str(marker or "").strip()
        if not value:
            continue
        if value.lower() in lowered:
            hits.append(value)
    return hits


@dataclass(frozen=True)
class ExpectationCoverageTool:
    name: str = "expectation_coverage"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        del context
        profile = state.get("agent_profile")
        plan = state.get("retrieval_plan")
        mode = ""
        if isinstance(plan, RetrievalPlan):
            mode = str(plan.mode or "").strip()
        expectations = list(getattr(profile, "expectations", []))
        if not expectations:
            return ToolResult(
                tool=self.name,
                ok=True,
                output={
                    "covered": [],
                    "missing": [],
                    "total_expectations": 0,
                    "coverage_ratio": 1.0,
                },
            )

        evidence = list(state.get("retrieved_documents") or [])
        if not evidence:
            return ToolResult(
                tool=self.name,
                ok=True,
                output={
                    "covered": [],
                    "missing": [
                        {
                            "id": str(item.id),
                            "description": str(item.description),
                            "missing_risk": str(item.missing_risk),
                            "severity": str(item.severity),
                            "reason": "no_retrieval_evidence",
                        }
                        for item in expectations
                    ],
                    "total_expectations": len(expectations),
                    "coverage_ratio": 0.0,
                },
            )

        covered: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []

        for expectation in expectations:
            applies_to_modes = [str(item or "").strip() for item in expectation.applies_to_modes]
            if applies_to_modes and mode and mode not in applies_to_modes:
                continue

            expectation_clauses = [
                str(item or "").strip() for item in expectation.clause_refs if item
            ]
            required_markers = [
                str(item or "").strip() for item in expectation.required_evidence_markers if item
            ]
            optional_markers = [
                str(item or "").strip() for item in expectation.optional_evidence_markers if item
            ]
            scopes = [str(item or "").strip() for item in expectation.scopes if item]

            matched_sources: list[str] = []
            matched_required: set[str] = set()
            matched_optional: set[str] = set()
            matched_clauses: set[str] = set()

            for item in evidence:
                source = str(getattr(item, "source", "") or "").strip()
                content = str(getattr(item, "content", "") or "").strip()
                row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
                if not isinstance(row, dict):
                    row = {"content": content, "metadata": {}}
                row_standard = extract_row_standard(item)
                if not _standard_allowed(row_standard, scopes):
                    continue

                content_blob = " ".join(content.split())
                row_clauses = _row_clause_refs(row)

                if expectation_clauses:
                    clause_hit = False
                    for clause in expectation_clauses:
                        clause_pattern = re.compile(rf"\b{re.escape(clause)}(?:\.\d+)*\b")
                        if clause_pattern.search(content_blob) or any(
                            _clause_ref_matches(clause, ref) for ref in row_clauses
                        ):
                            matched_clauses.add(clause)
                            clause_hit = True
                    if not clause_hit:
                        continue

                req_hits = _marker_hits(content_blob, required_markers)
                opt_hits = _marker_hits(content_blob, optional_markers)
                matched_required.update(req_hits)
                matched_optional.update(opt_hits)
                if source:
                    matched_sources.append(source)

            required_ok = bool(matched_required) if required_markers else True
            clauses_ok = (
                len(matched_clauses) >= max(1, len(expectation_clauses))
                if expectation_clauses
                else True
            )

            if required_ok and clauses_ok:
                covered.append(
                    {
                        "id": str(expectation.id),
                        "description": str(expectation.description),
                        "sources": sorted(set(matched_sources))[:8],
                        "matched_required_markers": sorted(matched_required),
                        "matched_optional_markers": sorted(matched_optional),
                        "matched_clause_refs": sorted(matched_clauses),
                    }
                )
                continue

            reason = ""
            if not clauses_ok:
                reason = "missing_clause_support"
            elif not required_ok:
                reason = "missing_required_markers"
            missing.append(
                {
                    "id": str(expectation.id),
                    "description": str(expectation.description),
                    "missing_risk": str(expectation.missing_risk),
                    "severity": str(expectation.severity),
                    "required_markers": required_markers,
                    "clause_refs": expectation_clauses,
                    "reason": reason,
                }
            )

        total = len(covered) + len(missing)
        coverage_ratio = round((len(covered) / total), 3) if total else 1.0
        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "covered": covered,
                "missing": missing,
                "total_expectations": total,
                "coverage_ratio": coverage_ratio,
            },
        )
