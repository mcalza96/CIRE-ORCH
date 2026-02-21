from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan
from app.cartridges.models import AgentProfile
from app.infrastructure.config import settings


def _clip(text: str, limit: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit].rstrip() + "..."


def _get_row_content_and_meta(item: EvidenceItem) -> tuple[str, dict[str, Any]]:
    row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
    if not isinstance(row, dict):
        return "", {}
    content = str(row.get("content") or "")
    meta_raw = row.get("metadata")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    return content, meta


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


def _row_mentions_scopes(item: EvidenceItem, scope_labels: list[str]) -> set[str]:
    content, meta = _get_row_content_and_meta(item)
    if not content and not meta:
        return set()
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
    return bool(req and cand and (cand == req or cand.startswith(f"{req}.")))


def _row_matches_clause(item: EvidenceItem, clause_refs: list[str]) -> bool:
    if not clause_refs:
        return False
    content, meta = _get_row_content_and_meta(item)
    if not content and not meta:
        return False

    values: list[str] = []
    for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
        val = str(meta.get(key) or "").strip()
        if val:
            values.append(val)

    refs_raw = meta.get("clause_refs")
    if isinstance(refs_raw, list):
        values.extend(str(v).strip() for v in refs_raw if isinstance(v, str) and str(v).strip())

    for ref in clause_refs:
        if re.search(rf"\b{re.escape(ref)}(?:\.\d+)*\b", content):
            return True
        if any(_clause_match(ref, candidate) for candidate in values):
            return True
    return False


def _snippet(text: str, limit: int = 240) -> str:
    raw = " ".join((text or "").split())
    return raw if len(raw) <= limit else raw[:limit].rstrip() + "..."


def _extract_subquestions(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\?+|\n+", raw):
        candidate = " ".join(part.split()).strip(" .:-")
        if len(candidate) < 18:
            continue
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _row_clause_label(item: EvidenceItem) -> str:
    _, meta = _get_row_content_and_meta(item)
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
        rows.append(f'{len(rows) + 1}) {label} | "{_snippet(item.content)}" | Fuente ({src})')
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
    _, meta = _get_row_content_and_meta(item)
    ts = (
        _safe_parse_iso8601(meta.get("updated_at"))
        or _safe_parse_iso8601(meta.get("source_updated_at"))
        or _safe_parse_iso8601(meta.get("created_at"))
    )
    return float(ts or 0.0)


def _append_missing_sources(text: str, items: list[EvidenceItem]) -> str:
    if re.search(r"\b[CR]\d+\b", text or ""):
        return text or ""
    sources: list[str] = []
    seen: set[str] = set()
    for item in items:
        src = (item.source or "").strip()
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    if sources:
        suffix = "Referencias revisadas: " + ", ".join(sources)
        text = (text or "").rstrip()
        return f"{text}\n\n{suffix}" if text else suffix
    return text or ""


def _scope_aliases(value: str) -> tuple[str, ...]:
    text = str(value or "").strip().upper()
    if not text:
        return ()
    aliases: list[str] = [text.casefold()]
    for match in re.findall(r"\b\d{3,6}\b", text):
        aliases.append(str(match).casefold())
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        ordered.append(alias)
    return tuple(ordered)


def _item_matches_scope(item: EvidenceItem, scope: str) -> bool:
    aliases = _scope_aliases(scope)
    if not aliases:
        return False
    content, meta = _get_row_content_and_meta(item)
    blob = "\n".join(
        part
        for part in [
            str(content or ""),
            json.dumps(meta or {}, default=str, ensure_ascii=True),
        ]
        if part
    ).casefold()
    if not blob:
        return False
    for alias in aliases:
        if alias in blob:
            return True
    return False


def _balance_evidence_by_scope(
    *,
    items: list[EvidenceItem],
    requested_scopes: tuple[str, ...],
    max_items: int,
) -> list[EvidenceItem]:
    if not bool(getattr(settings, "ORCH_SCOPE_BALANCE_FINAL_ENABLED", True)):
        return items[: max(1, max_items)]
    scopes = [str(scope or "").strip().upper() for scope in requested_scopes if str(scope).strip()]
    if len(scopes) < 2:
        return items[: max(1, max_items)]

    per_scope: dict[str, list[EvidenceItem]] = {scope: [] for scope in scopes}
    leftovers: list[EvidenceItem] = []
    for item in items:
        matched = False
        for scope in scopes:
            if _item_matches_scope(item, scope):
                per_scope[scope].append(item)
                matched = True
                break
        if not matched:
            leftovers.append(item)

    min_per_scope = max(1, int(getattr(settings, "ORCH_SCOPE_BALANCE_MIN_PER_SCOPE", 2) or 2))
    selected: list[EvidenceItem] = []
    seen_sources: set[str] = set()

    def _add(item: EvidenceItem) -> None:
        source = str(item.source or "").strip()
        key = source or str(id(item))
        if key in seen_sources:
            return
        seen_sources.add(key)
        selected.append(item)

    for scope in scopes:
        for item in per_scope.get(scope, [])[:min_per_scope]:
            _add(item)

    cursor = {scope: min(len(per_scope.get(scope, [])), min_per_scope) for scope in scopes}
    while len(selected) < max(1, max_items):
        progressed = False
        for scope in scopes:
            bucket = per_scope.get(scope, [])
            idx = cursor.get(scope, 0)
            if idx >= len(bucket):
                continue
            _add(bucket[idx])
            cursor[scope] = idx + 1
            progressed = True
            if len(selected) >= max(1, max_items):
                break
        if not progressed:
            break

    for item in leftovers:
        if len(selected) >= max(1, max_items):
            break
        _add(item)

    if len(selected) < max(1, max_items):
        for item in items:
            if len(selected) >= max(1, max_items):
                break
            _add(item)

    return selected[: max(1, max_items)]


class GroundedAnswerAdapter:
    def __init__(self, service: GroundedAnswerService):
        self.service = service

    def _build_generation_query(
        self, query: str, plan: RetrievalPlan, literal_min_items: int
    ) -> str:
        if plan.require_literal_evidence and literal_min_items > 1:
            return (
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
                return (
                    query + "\n\n[INSTRUCCION INTERNA] La consulta contiene multiples preguntas. "
                    "Responde cada una con subtitulo propio, en el mismo orden, y cita evidencia por seccion:\n"
                    + numbered
                )
        return query

    def _apply_post_generation_guardrails(
        self,
        text: str,
        ordered_items: list[EvidenceItem],
        plan: RetrievalPlan,
        agent_profile: AgentProfile | None,
        cross_scope_mode: bool,
        clause_items: list[EvidenceItem],
        literal_min_items: int,
        clause_refs_count: int,
    ) -> str:
        if not ordered_items and cross_scope_mode:
            lines = [
                f"**{scope}**: No encontrado explicitamente en el contexto recuperado."
                for scope in plan.requested_standards
            ]
            return "\n\n".join(lines)

        requested_scopes = list(plan.requested_standards or ())
        candidate_scopes: list[str] = []
        if agent_profile is not None:
            candidate_scopes.extend(list(agent_profile.router.scope_hints.keys()))
            candidate_scopes.extend(list(agent_profile.domain_entities))
        candidate_scopes.extend(requested_scopes)

        mentioned_scopes = _extract_scope_labels(text, candidate_scopes)
        scopes = requested_scopes if len(requested_scopes) >= 2 else mentioned_scopes

        if len(scopes) >= 2 and not plan.require_literal_evidence:
            bridges = sum(1 for ev in ordered_items if len(_row_mentions_scopes(ev, scopes)) >= 2)
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
            text = _append_missing_sources(text, ordered_items)
            markers = set(re.findall(r"\b[CR]\d+\b", text or ""))
            if literal_min_items > 1 and len(markers) < 2 and len(clause_items) >= 2:
                text = _render_literal_rows(clause_items, max_rows=2)

        low_text = (text or "").strip().lower()
        if low_text.startswith("no encontrado explicitamente") or low_text.startswith(
            "no encuentro evidencia suficiente"
        ):
            if ordered_items:
                preferred = (
                    clause_items
                    if (plan.require_literal_evidence and clause_items)
                    else ordered_items
                )
                max_rows = 3 if clause_refs_count == 0 else 2
                rendered = _render_literal_rows(preferred, max_rows=max_rows)
                if rendered:
                    text = rendered

        if ordered_items and not plan.require_literal_evidence:
            # Transversal citation contract
            text = _append_missing_sources(text, ordered_items)

        return text

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
        **kwargs,
    ) -> AnswerDraft:
        ordered_items = [*summaries, *chunks]
        cross_scope_mode = len(plan.requested_standards) >= 2 and not plan.require_literal_evidence
        if cross_scope_mode:
            ordered_items = sorted(ordered_items, key=_recency_key, reverse=True)

        max_ctx = 28 if cross_scope_mode else (14 if plan.require_literal_evidence else 18)
        generation_items = (
            _balance_evidence_by_scope(
                items=ordered_items,
                requested_scopes=tuple(plan.requested_standards or ()),
                max_items=max_ctx,
            )
            if cross_scope_mode
            else ordered_items[: max(1, max_ctx)]
        )

        labeled: list[str] = []
        for item in generation_items:
            content = (item.content or "").strip()
            if content:
                source = (item.source or "").strip() or "C1"
                labeled.append(f"[{source}] {_clip(content, 900)}")

        clause_refs = _extract_clause_refs(query)
        clause_items = [item for item in generation_items if _row_matches_clause(item, clause_refs)]
        literal_min_items = 2 if plan.require_literal_evidence and len(clause_items) >= 2 else 1

        query_for_generation = self._build_generation_query(query, plan, literal_min_items)

        structured_context_parts = []
        if working_memory:
            structured_context_parts.append(
                f"WORKING_MEMORY:\n{json.dumps(working_memory, indent=2, sort_keys=True, default=str)}"
            )
        if partial_answers:
            structured_context_parts.append(
                f"PARTIAL_ANSWERS:\n{json.dumps(partial_answers, indent=2, sort_keys=True, default=str)}"
            )
        structured_context = (
            "\n\n".join(structured_context_parts) if structured_context_parts else None
        )

        text = await self.service.generate_answer(
            query=query_for_generation,
            context_chunks=labeled,
            agent_profile=agent_profile,
            mode=plan.mode,
            require_literal_evidence=bool(plan.require_literal_evidence),
            structured_context=structured_context,
            max_chunks=max_ctx,
        )

        text = self._apply_post_generation_guardrails(
            text=text,
            ordered_items=generation_items,
            plan=plan,
            agent_profile=agent_profile,
            cross_scope_mode=cross_scope_mode,
            clause_items=clause_items,
            literal_min_items=literal_min_items,
            clause_refs_count=len(clause_refs),
        )

        return AnswerDraft(text=text, mode=plan.mode, evidence=generation_items)
