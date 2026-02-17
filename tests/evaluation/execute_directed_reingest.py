#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_PLAN = Path(__file__).with_name("reports") / "directed_reingest_plan_iterC.json"
DEFAULT_OUT = Path(__file__).with_name("reports") / "directed_reingest_execution.json"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _standard_tokens(standard: str) -> list[str]:
    digits = "".join(ch for ch in standard if ch.isdigit())
    if not digits:
        return [_normalize(standard)]
    return [f"iso {digits}", f"iso-{digits}", f"iso{digits}", digits]


def _flatten_doc_text(doc: dict[str, Any]) -> str:
    bits: list[str] = []
    bits.append(str(doc.get("filename") or ""))
    bits.append(str(doc.get("title") or ""))
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        bits.append(json.dumps(metadata, ensure_ascii=True, sort_keys=True))
    return _normalize(" ".join(bits))


def _extract_storage_path(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        top = str(metadata.get("storage_path") or "").strip()
        if top:
            return top
        nested = metadata.get("metadata")
        if isinstance(nested, dict):
            nested_path = str(nested.get("storage_path") or "").strip()
            if nested_path:
                return nested_path
    return ""


def _extract_collection_context(doc: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    metadata = doc.get("metadata")
    nested = None
    if isinstance(metadata, dict):
        nested = metadata.get("metadata")

    for key in ("collection_id", "collection_key", "collection_name"):
        val = ""
        if isinstance(metadata, dict):
            val = str(metadata.get(key) or "").strip()
        if not val and isinstance(nested, dict):
            val = str(nested.get(key) or "").strip()
        if val:
            out[key] = val
    return out


@dataclass(frozen=True)
class Candidate:
    source_doc_id: str
    filename: str
    storage_path: str
    standard: str
    score: int
    matched_clauses: list[str]
    collection_context: dict[str, str]


def _score_doc_for_standard(
    doc_text: str, standard: str, clauses: list[str]
) -> tuple[int, list[str]]:
    tokens = _standard_tokens(standard)
    standard_hits = sum(1 for token in tokens if token and token in doc_text)
    matched_clauses = [cl for cl in clauses if str(cl).strip() and str(cl).strip() in doc_text]

    score = 0
    if standard_hits > 0:
        score += 100
        score += standard_hits * 10
    score += len(matched_clauses) * 8
    return score, matched_clauses


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


async def _fetch_documents(
    *,
    rag_base_url: str,
    tenant_id: str,
    service_secret: str,
    bearer_token: str,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'httpx'.") from exc

    headers: dict[str, str] = {"X-Tenant-ID": tenant_id}
    secret = str(service_secret or "").strip()
    token = str(bearer_token or "").strip()
    if secret:
        headers["X-Service-Secret"] = secret
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{rag_base_url.rstrip('/')}/api/v1/ingestion/documents"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        response = await client.get(url, params={"limit": int(limit)})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected documents payload shape")
        return [row for row in payload if isinstance(row, dict)]


async def _post_institutional_ingest(
    *,
    rag_base_url: str,
    tenant_id: str,
    service_secret: str,
    bearer_token: str,
    file_path: str,
    payload_metadata: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'httpx'.") from exc

    headers: dict[str, str] = {"X-Tenant-ID": tenant_id}
    secret = str(service_secret or "").strip()
    token = str(bearer_token or "").strip()
    if secret:
        headers["X-Service-Secret"] = secret
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {
        "tenant_id": tenant_id,
        "file_path": file_path,
        "document_id": str(uuid4()),
        "metadata": payload_metadata,
    }

    timeout = httpx.Timeout(45.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        response = await client.post(
            f"{rag_base_url.rstrip('/')}/api/v1/ingestion/institutional",
            json=body,
        )
        data: dict[str, Any] = {}
        try:
            raw = response.json()
            if isinstance(raw, dict):
                data = raw
            else:
                data = {"raw": raw}
        except Exception:
            data = {"raw_text": response.text[:1200]}
        return response.status_code, data


async def main() -> int:
    parser = argparse.ArgumentParser(description="Executes directed reingest actions from a plan")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rag-base-url", default="http://localhost:8000")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--service-secret", default="")
    parser.add_argument("--bearer-token", default="")
    parser.add_argument("--documents-json", type=Path)
    parser.add_argument("--export-documents-json", type=Path)
    parser.add_argument("--documents-limit", type=int, default=500)
    parser.add_argument("--top-k-per-standard", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    plan_payload = _load_json(args.plan)
    by_standard = plan_payload.get("by_standard") if isinstance(plan_payload, dict) else {}
    priority = plan_payload.get("priority") if isinstance(plan_payload, dict) else []
    if not isinstance(by_standard, dict) or not by_standard:
        raise SystemExit("Plan has no by_standard targets")

    docs: list[dict[str, Any]]
    if args.documents_json:
        rows = _load_json(args.documents_json)
        if not isinstance(rows, list):
            raise SystemExit("--documents-json must contain a JSON array")
        docs = [row for row in rows if isinstance(row, dict)]
    else:
        docs = await _fetch_documents(
            rag_base_url=args.rag_base_url,
            tenant_id=str(args.tenant_id).strip(),
            service_secret=args.service_secret,
            bearer_token=args.bearer_token,
            limit=max(1, int(args.documents_limit)),
        )

    if args.export_documents_json:
        args.export_documents_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_documents_json.write_text(
            json.dumps(docs, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    docs_index: list[tuple[dict[str, Any], str]] = [(doc, _flatten_doc_text(doc)) for doc in docs]

    priority_order: list[str] = []
    if isinstance(priority, list) and priority:
        for item in priority:
            if not isinstance(item, dict):
                continue
            std = str(item.get("standard") or "").strip()
            if std and std in by_standard:
                priority_order.append(std)
    for std in by_standard.keys():
        if std not in priority_order:
            priority_order.append(std)

    selected_actions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    top_k = max(1, int(args.top_k_per_standard))
    for standard in priority_order:
        info = by_standard.get(standard)
        if not isinstance(info, dict):
            continue
        clauses = [str(item).strip() for item in (info.get("clauses") or []) if str(item).strip()]

        ranked: list[Candidate] = []
        for doc, doc_text in docs_index:
            score, matched_clauses = _score_doc_for_standard(doc_text, standard, clauses)
            if score <= 0:
                continue
            storage_path = _extract_storage_path(doc)
            if not storage_path:
                continue
            ranked.append(
                Candidate(
                    source_doc_id=str(doc.get("id") or "").strip(),
                    filename=str(doc.get("filename") or "").strip(),
                    storage_path=storage_path,
                    standard=standard,
                    score=score,
                    matched_clauses=matched_clauses,
                    collection_context=_extract_collection_context(doc),
                )
            )

        ranked.sort(key=lambda c: (c.score, len(c.matched_clauses)), reverse=True)
        if not ranked:
            unresolved.append({"standard": standard, "reason": "no_matching_document"})
            continue

        for cand in ranked[:top_k]:
            selected_actions.append(
                {
                    "standard": standard,
                    "source_doc_id": cand.source_doc_id,
                    "filename": cand.filename,
                    "storage_path": cand.storage_path,
                    "score": cand.score,
                    "matched_clauses": cand.matched_clauses,
                    "collection_context": cand.collection_context,
                }
            )

    executions: list[dict[str, Any]] = []
    if args.apply:
        for action in selected_actions:
            standard = str(action.get("standard") or "").strip()
            payload_metadata = {
                "title": f"directed_reingest_{standard.lower().replace(' ', '_')}",
                "institution_id": str(args.tenant_id).strip(),
                **(action.get("collection_context") or {}),
                "metadata": {
                    "directed_reingest": True,
                    "source_report": str(plan_payload.get("source_report") or ""),
                    "standard": standard,
                    "matched_clauses": action.get("matched_clauses", []),
                    "source_doc_id": action.get("source_doc_id"),
                    **(action.get("collection_context") or {}),
                },
            }
            status_code, payload = await _post_institutional_ingest(
                rag_base_url=args.rag_base_url,
                tenant_id=str(args.tenant_id).strip(),
                service_secret=args.service_secret,
                bearer_token=args.bearer_token,
                file_path=str(action.get("storage_path") or ""),
                payload_metadata=payload_metadata,
            )
            executions.append(
                {
                    "standard": standard,
                    "storage_path": action.get("storage_path"),
                    "status_code": int(status_code),
                    "ok": int(status_code) == 200,
                    "response": payload,
                }
            )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan),
        "rag_base_url": args.rag_base_url,
        "tenant_id": str(args.tenant_id).strip(),
        "mode": "apply" if args.apply else "dry_run",
        "documents_count": len(docs),
        "selected_actions": selected_actions,
        "unresolved": unresolved,
        "executions": executions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "mode": output["mode"],
                "documents_count": output["documents_count"],
                "selected_actions": len(selected_actions),
                "unresolved": len(unresolved),
                "executed": len(executions),
                "ok": sum(1 for row in executions if row.get("ok")),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
