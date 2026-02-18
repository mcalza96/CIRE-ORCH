from __future__ import annotations

import re
from typing import Any

from app.cartridges.models import AgentProfile


_ISO_RE = re.compile(r"\bISO\s*[-:]?\s*(\d{4,5})\b", re.IGNORECASE)
_CLAUSE_ID_RE = re.compile(r"\[\s*CLAUSE_ID\s*:\s*([0-9]+(?:\.[0-9]+)+)\s*\]", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"\b(?:cl(?:a|á)usula\s*)?([0-9]+(?:\.[0-9]+)+)\b", re.IGNORECASE)


def _compact_text(text: str, *, limit: int = 220) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "..."


def _extract_standard(row_meta: dict[str, Any], content: str) -> str:
    for field in ("source_standard", "standard", "scope"):
        value = str(row_meta.get(field) or "").strip()
        if value:
            return value
    m = _ISO_RE.search(content)
    if m:
        return f"ISO {m.group(1)}"
    return ""


def _extract_clause(row_meta: dict[str, Any], content: str) -> str:
    for field in ("clause_id", "clause_ref", "clause", "clause_anchor"):
        value = str(row_meta.get(field) or "").strip()
        if value:
            return value
    m = _CLAUSE_ID_RE.search(content)
    if m:
        return m.group(1)
    m2 = _CLAUSE_RE.search(content)
    if m2:
        return m2.group(1)
    return ""


def _is_noise(content: str, filters: list[str]) -> bool:
    lowered = content.lower()
    for token in filters:
        value = str(token or "").strip().lower()
        if value and value in lowered:
            return True
    return False


def _safe_render(template: str, payload: dict[str, Any]) -> str:
    out = str(template or "").strip()
    if not out:
        out = '{id} | {standard} | clausula {clause_id} | "{snippet}"'
    for key, value in payload.items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out


def build_citation_bundle(
    *,
    answer_text: str,
    evidence: list[Any],
    profile: AgentProfile | None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    used_markers = {
        marker.upper()
        for marker in re.findall(r"\b[CR]\d+\b", str(answer_text or ""), flags=re.IGNORECASE)
    }
    synthesis = profile.synthesis if profile is not None else None
    required_fields = (
        list(synthesis.citation_required_fields)
        if synthesis is not None and synthesis.citation_required_fields
        else ["id", "standard", "clause_id", "quote"]
    )
    render_template = (
        str(synthesis.citation_render_template).strip()
        if synthesis is not None and synthesis.citation_render_template
        else '{id} | {standard} | clausula {clause_id} | "{snippet}"'
    )
    noise_filters = (
        list(synthesis.citation_noise_filters)
        if synthesis is not None and synthesis.citation_noise_filters
        else ["indice", "prólogo", "traducción oficial", "official translation"]
    )

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    discarded_noise = 0
    structured_count = 0

    for item in evidence:
        source = str(getattr(item, "source", "") or "").strip()
        if not source:
            continue
        key = source.upper()
        if key in seen:
            continue
        seen.add(key)

        score = getattr(item, "score", None)
        content = str(getattr(item, "content", "") or "")
        metadata = getattr(item, "metadata", None)
        row = metadata.get("row") if isinstance(metadata, dict) else None
        row_meta_raw = row.get("metadata") if isinstance(row, dict) else None
        row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}

        standard = _extract_standard(row_meta, content)
        clause_id = _extract_clause(row_meta, content)
        snippet = _compact_text(content)
        noise = _is_noise(content, noise_filters)

        payload = {
            "id": source,
            "standard": standard or "N/A",
            "clause_id": clause_id or "N/A",
            "snippet": snippet,
            "quote": snippet,
        }
        missing_fields: list[str] = []
        for field in required_fields:
            value = str(payload.get(field) or "").strip()
            if not value or value == "N/A":
                missing_fields.append(field)

        used = key in used_markers
        if not missing_fields and not noise:
            structured_count += 1
        if noise:
            discarded_noise += 1

        details.append(
            {
                "id": source,
                "standard": standard,
                "clause": clause_id,
                "score": float(score) if isinstance(score, (int, float)) else None,
                "snippet": snippet,
                "used_in_answer": used,
                "missing_fields": missing_fields,
                "noise": noise,
                "rendered": _safe_render(render_template, payload),
            }
        )

    details.sort(
        key=lambda item: (
            bool(item.get("noise", False)),
            not bool(item.get("used_in_answer", False)),
            len(item.get("missing_fields") or []),
            -float(item.get("score") or 0.0),
            str(item.get("id") or ""),
        )
    )
    citations = [
        str(item.get("id") or "")
        for item in details
        if str(item.get("id") or "") and not bool(item.get("noise", False))
    ]
    ratio = float(structured_count / max(1, len(details)))
    quality = {
        "schema_version": (
            str(synthesis.citation_schema_version).strip()
            if synthesis is not None and synthesis.citation_schema_version
            else "v1"
        ),
        "total": len(details),
        "structured_count": structured_count,
        "structured_ratio": round(ratio, 4),
        "discarded_noise": discarded_noise,
        "required_fields": required_fields,
        "min_structured_citation_ratio": (
            float(synthesis.min_structured_citation_ratio) if synthesis is not None else 0.5
        ),
    }
    return citations, details, quality
