from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan
from app.cartridges.models import AgentProfile


def _clip(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "..."


def _extract_scope_labels(text: str, candidates: list[str]) -> list[str]:
    source = (text or "").lower()
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        label = (candidate or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in source and key not in seen:
            seen.add(key)
            ordered.append(label)
    return ordered


def _row_mentions_scopes(row: dict[str, Any], scope_labels: list[str]) -> set[str]:
    content = str(row.get("content") or "")
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    blob = (content + "\n" + json.dumps(meta, default=str, ensure_ascii=True)).lower()
    present: set[str] = set()
    for label in scope_labels:
        key = (label or "").strip().lower()
        if key and key in blob:
            present.add(label)
    return present


def _extract_clause_refs(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in re.findall(r"\b\d+(?:\.\d+)+\b", text or ""):
        value = str(match).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _clause_match(requested: str, candidate: str) -> bool:
    req = str(requested or "").strip()
    cand = str(candidate or "").strip()
    if not req or not cand:
        return False
    return cand == req or cand.startswith(f"{req}.")


def _row_matches_clause(item: EvidenceItem, clause_refs: list[str]) -> bool:
    if not clause_refs:
        return False
    row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
    if not isinstance(row, dict):
        return False
    content = str(row.get("content") or "")
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    values: list[str] = []
    for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
        val = str(meta.get(key) or "").strip()
        if val:
            values.append(val)
    refs_raw = meta.get("clause_refs")
    if isinstance(refs_raw, list):
        values.extend(
            str(v).strip() for v in refs_raw if isinstance(v, str) and str(v).strip()
        )
    for ref in clause_refs:
        if re.search(rf"\b{re.escape(ref)}(?:\.\d+)*\b", content):
            return True
        if any(_clause_match(ref, candidate) for candidate in values):
            return True
    return False


def _snippet(text: str, limit: int = 240) -> str:
    raw = " ".join((text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "..."


def _extract_subquestions(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"\?+|\n+", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = " ".join(part.split()).strip(" .:-")
        if len(candidate) < 18:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _row_clause_label(item: EvidenceItem) -> str:
    row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
    if not isinstance(row, dict):
        return ""
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return ""


def _render_literal_rows(items: list[EvidenceItem], *, max_rows: int) -> str:
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        src = str(item.source or "").strip()
        if not src or src in seen:
            continue
        seen.add(src)
        clause = _row_clause_label(item)
        label = f"Clausula {clause}" if clause else "Afirmacion"
        rows.append(
            f'{len(rows) + 1}) {label} | "{_snippet(item.content)}" | Fuente ({src})'
        )
        if len(rows) >= max_rows:
            break
    return "\n".join(rows)


def _safe_parse_iso8601(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _recency_key(item: EvidenceItem) -> float:
    row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
    if not isinstance(row, dict):
        return 0.0
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    ts = (
        _safe_parse_iso8601(meta.get("updated_at"))
        or _safe_parse_iso8601(meta.get("source_updated_at"))
        or _safe_parse_iso8601(meta.get("created_at"))
    )
    return float(ts or 0.0)


class GroundedAnswerAdapter:
    def __init__(self, service: GroundedAnswerService):
        self.service = service

    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
        working_memory: dict[str, Any] | None = None,
        partial_answers: list[dict[str, Any]] | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> AnswerDraft:
        # IMPORTANT: Literal modes are validated against explicit C#/R# markers.
        # Always pass the LLM a context that includes those markers, otherwise the
        # validator will (correctly) reject "literal" answers as ungrounded.
        del scope_label

        ordered_items = [*summaries, *chunks]
        cross_scope_mode = len(plan.requested_standards) >= 2 and not plan.require_literal_evidence
        if cross_scope_mode:
            # Best-effort recency: if timestamps exist in metadata, prefer more recent evidence.
            ordered_items = sorted(ordered_items, key=_recency_key, reverse=True)

        labeled: list[str] = []
        for item in ordered_items:
            content = (item.content or "").strip()
            if not content:
                continue
            source = (item.source or "").strip() or "C1"
            labeled.append(f"[{source}] {_clip(content, 900)}")

        # Use more context for multi-standard interpretive questions.
        if cross_scope_mode:
            max_ctx = 28
        elif plan.require_literal_evidence:
            max_ctx = 14
        else:
            max_ctx = 18

        clause_refs = _extract_clause_refs(query)
        clause_items = [item for item in ordered_items if _row_matches_clause(item, clause_refs)]
        literal_min_items = 2 if plan.require_literal_evidence and len(clause_items) >= 2 else 1

        query_for_generation = query
        if plan.require_literal_evidence and literal_min_items > 1:
            query_for_generation = (
                query
                + "\n\n[INSTRUCCION INTERNA] Si hay evidencia suficiente, responde con al menos 2 "
                "afirmaciones literales distintas de la misma clausula, cada una con su fuente C#/R#."
            )
        elif not plan.require_literal_evidence:
            subquestions = _extract_subquestions(query)
            if len(subquestions) >= 2:
                numbered = "\n".join(
                    f"{idx + 1}) {item}" for idx, item in enumerate(subquestions[:4])
                )
                query_for_generation = (
                    query + "\n\n[INSTRUCCION INTERNA] La consulta contiene multiples preguntas. "
                    "Responde cada una con subtitulo propio, en el mismo orden, y cita evidencia por seccion:\n"
                    + numbered
                )

        structured_context_parts = []
        if working_memory:
            structured_context_parts.append(f"WORKING_MEMORY:\n{json.dumps(working_memory, indent=2, sort_keys=True, default=str)}")
        if partial_answers:
            structured_context_parts.append(f"PARTIAL_ANSWERS:\n{json.dumps(partial_answers, indent=2, sort_keys=True, default=str)}")
        structured_context = "\n\n".join(structured_context_parts) if structured_context_parts else None

        text = await self.service.generate_answer(
            query=query_for_generation,
            context_chunks=labeled,
            agent_profile=agent_profile,
            mode=plan.mode,
            require_literal_evidence=bool(plan.require_literal_evidence),
            structured_context=structured_context,
            max_chunks=max_ctx,
        )

        if not ordered_items and cross_scope_mode:
            lines = [
                f"**{scope}**: No encontrado explicitamente en el contexto recuperado."
                for scope in plan.requested_standards
            ]
            text = "\n\n".join(lines)

        # Guardrail: if the answer ties multiple scopes together but evidence has no explicit bridge,
        # enforce transparent language (inference vs direct citation).
        requested_scopes = list(plan.requested_standards or ())
        candidate_scopes: list[str] = []
        if agent_profile is not None:
            candidate_scopes.extend(list(agent_profile.router.scope_hints.keys()))
            candidate_scopes.extend(list(agent_profile.domain_entities))
        candidate_scopes.extend(requested_scopes)

        mentioned_scopes = _extract_scope_labels(text, candidate_scopes)
        scopes = requested_scopes if len(requested_scopes) >= 2 else mentioned_scopes
        if len(scopes) >= 2 and not plan.require_literal_evidence:
            bridges = 0
            for ev in ordered_items:
                row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else None
                if not isinstance(row, dict):
                    continue
                present = _row_mentions_scopes(row, scopes)
                if len(present) >= 2:
                    bridges += 1
                    break
            already_disclosed = any(
                tok in (text or "").lower()
                for tok in ("inferencia", "interpret", "no encontrado explicitamente")
            )
            if bridges == 0 and not already_disclosed:
                note = (
                    "Nota de trazabilidad: La relacion entre fuentes se presenta como inferencia basada en "
                    "evidencias separadas; no hay un fragmento unico que las vincule explicitamente."
                )
                text = f"{note}\n\n{text}" if (text or "").strip() else note
        if plan.require_literal_evidence:
            # Defense-in-depth: if the provider ignores instructions and returns no markers,
            # append the reviewed references so the validator can trace the answer.
            if not re.search(r"\b[CR]\d+\b", text or ""):
                sources: list[str] = []
                seen: set[str] = set()
                for item in [*summaries, *chunks]:
                    src = (item.source or "").strip()
                    if not src or src in seen:
                        continue
                    seen.add(src)
                    sources.append(src)
                if sources:
                    suffix = "Referencias revisadas: " + ", ".join(sources)
                    text = (text or "").rstrip()
                    text = f"{text}\n\n{suffix}" if text else suffix

            # Ensure literal outputs include at least two grounded rows when available.
            markers = set(re.findall(r"\b[CR]\d+\b", text or ""))
            if literal_min_items > 1 and len(markers) < 2 and len(clause_items) >= 2:
                text = _render_literal_rows(clause_items, max_rows=2)

            low_text = (text or "").strip().lower()
            looks_fallback = low_text.startswith(
                "no encontrado explicitamente"
            ) or low_text.startswith("no encuentro evidencia suficiente")
            if looks_fallback and ordered_items:
                preferred = clause_items if clause_items else ordered_items
                max_rows = 3 if len(clause_refs) == 0 else 2
                rendered = _render_literal_rows(preferred, max_rows=max_rows)
                if rendered:
                    text = rendered
        else:
            low_text = (text or "").strip().lower()
            looks_fallback = low_text.startswith(
                "no encontrado explicitamente"
            ) or low_text.startswith("no encuentro evidencia suficiente")
            if looks_fallback and ordered_items:
                rendered = _render_literal_rows(ordered_items, max_rows=3)
                if rendered:
                    text = rendered

        # Transversal citation contract: when evidence exists, ensure explicit markers C#/R#.
        if [*chunks, *summaries] and not re.search(r"\b[CR]\d+\b", text or ""):
            sources: list[str] = []
            seen: set[str] = set()
            for item in [*summaries, *chunks]:
                src = (item.source or "").strip()
                if not src or src in seen:
                    continue
                seen.add(src)
                sources.append(src)
            if sources:
                suffix = "Referencias revisadas: " + ", ".join(sources)
                text = (text or "").rstrip()
                text = f"{text}\n\n{suffix}" if text else suffix
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])
