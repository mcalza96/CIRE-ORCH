from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

from app.profiles.models import AgentProfile
from app.agent.types.models import AnswerDraft, EvidenceItem, RetrievalPlan, ValidationResult
from app.agent.policies import extract_requested_scopes
from app.infrastructure.config import settings


def _extract_keywords(query: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\.]{3,}", query.lower()))
    stop = {
        "para",
        "como",
        "cómo",
        "donde",
        "dónde",
        "sobre",
        "entre",
        "respecto",
        "tiene",
        "tienen",
        "debe",
        "deben",
        "norma",
        "normas",
        "alcance",
        "referencia",
        "requisitos",
        "pregunta",
        "diferencia",
        "difiere",
        "ambas",
    }
    return {t for t in terms if t not in stop}


def _extract_clause_refs(query: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)+\b", (query or "")))


def _clause_ref_matches(requested: str, candidate: str) -> bool:
    req = str(requested or "").strip()
    cand = str(candidate or "").strip()
    if not req or not cand:
        return False
    return cand == req or cand.startswith(f"{req}.")


def _extract_metadata_clause_refs(row: dict[str, Any]) -> set[str]:
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    refs_raw = meta.get("clause_refs")
    refs = {
        str(value).strip()
        for value in (refs_raw if isinstance(refs_raw, list) else [])
        if isinstance(value, str) and value.strip()
    }
    clause_anchor = str(meta.get("clause_anchor") or "").strip()
    if clause_anchor:
        refs.add(clause_anchor)
    for key in ("clause_id", "clause_ref", "clause"):
        value = str(meta.get(key) or "").strip()
        if value:
            refs.add(value)
    return refs


def _semantic_clause_match(
    *,
    query: str,
    row: dict[str, Any],
    requested_upper: set[str],
) -> bool:
    if not settings.QA_LITERAL_SEMANTIC_FALLBACK_ENABLED:
        return False

    row_scope = _extract_row_standard(row)
    if requested_upper and row_scope and not any(target in row_scope for target in requested_upper):
        return False

    content = str(row.get("content") or "")
    similarity = float(row.get("similarity") or 0.0)
    refs = _extract_metadata_clause_refs(row)
    meta_raw = row.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    clause_title = str(meta.get("clause_title") or "")

    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return False

    evidence_text = f"{content}\n{clause_title}\n{' '.join(refs)}".lower()
    overlap = sum(1 for kw in query_keywords if kw in evidence_text)

    if overlap < max(1, int(settings.QA_LITERAL_SEMANTIC_MIN_KEYWORD_OVERLAP)):
        return False

    return similarity >= float(settings.QA_LITERAL_SEMANTIC_MIN_SIMILARITY)


def _extract_row_standard(row: dict[str, Any]) -> str:
    meta_raw = row.get("metadata")
    metadata: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    candidates = [
        metadata.get("source_standard"),
        metadata.get("standard"),
        metadata.get("scope"),
        metadata.get("norma"),
        row.get("source_standard"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _row_matches_standards(row: dict[str, Any], standards: list[str]) -> bool:
    if not standards:
        return True
    canonical = {s.upper() for s in standards}
    row_standard = _extract_row_standard(row)
    if row_standard:
        if any(s in row_standard for s in canonical):
            return True
        return False

    content = str(row.get("content") or "").upper()
    return any(s in content for s in canonical)


def _scope_matches(row_scope: str, requested_scope: str) -> bool:
    row_text = str(row_scope or "").strip().upper()
    req_text = str(requested_scope or "").strip().upper()
    if not row_text or not req_text:
        return False
    if req_text in row_text or row_text in req_text:
        return True
    req_digits = re.findall(r"\b\d{3,6}\b", req_text)
    if req_digits and any(digit in row_text for digit in req_digits):
        return True
    return False


def _rerank_for_literal(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keywords = _extract_keywords(query)
    clause_refs = _extract_clause_refs(query)
    requested_standards = [item.upper() for item in extract_requested_scopes(query)]
    if not keywords and not clause_refs:
        return rows

    def score(row: dict[str, Any]) -> tuple[int, float]:
        content = str(row.get("content") or "").lower()
        meta_raw = row.get("metadata")
        meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
        meta_clause_refs = _extract_metadata_clause_refs(row)

        overlap = sum(1 for kw in keywords if kw in content)
        clause_boost = sum(2 for ref in clause_refs if ref in content)
        clause_meta_boost = sum(4 for ref in clause_refs if ref in meta_clause_refs)
        standard_boost = 0
        if requested_standards:
            row_standard = _extract_row_standard(row)
            if row_standard and any(target in row_standard for target in requested_standards):
                standard_boost = 2

        similarity = float(row.get("similarity") or 0.0)
        return (overlap + clause_boost + clause_meta_boost + standard_boost, similarity)

    return sorted(rows, key=score, reverse=True)


@dataclass
class RetrievalToolsAdapter:
    tools: Any
    collection_name: str | None = None
    allowed_source_ids: set[str] | None = None

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        del user_id
        requested_standards = list(extract_requested_scopes(query))
        strict_scope = plan.require_literal_evidence and bool(requested_standards)

        scope_context: dict[str, Any] = {"type": "institutional", "tenant_id": tenant_id}
        scoped_filters: dict[str, Any] = {}
        if collection_id:
            scoped_filters["collection_id"] = collection_id
        elif self.collection_name:
            scoped_filters["collection_name"] = self.collection_name

        if requested_standards:
            scoped_filters["source_standards"] = requested_standards
            if len(requested_standards) == 1:
                scoped_filters["source_standard"] = requested_standards[0]

        if scoped_filters:
            scope_context["filters"] = scoped_filters

        rows = await self.tools.retrieve(
            query=query,
            scope_context=scope_context,
            k=plan.chunk_k,
            fetch_k=plan.chunk_fetch_k,
            enable_reranking=True,
        )

        if self.allowed_source_ids:
            rows = [
                r
                for r in rows
                if str(r.get("source_id") or r.get("id") or "") in self.allowed_source_ids
            ]

        filtered_by_scope = [r for r in rows if _row_matches_standards(r, requested_standards)]
        if strict_scope and bool(getattr(settings, "SCOPE_STRICT_FILTERING", False)):
            rows = filtered_by_scope
        elif strict_scope and filtered_by_scope:
            rows = filtered_by_scope
        elif not strict_scope:
            rows = filtered_by_scope or rows

        if plan.require_literal_evidence:
            rows = _rerank_for_literal(query, rows)

        return [
            EvidenceItem(
                source=str(row.get("id") or f"chunk-{i}"),
                content=str(row.get("content") or "").strip(),
                score=float(row.get("similarity") or 0.0),
                metadata={"row": row},
            )
            for i, row in enumerate(rows)
            if row.get("content")
        ]

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        del user_id
        requested_standards = list(extract_requested_scopes(query))
        strict_scope = plan.require_literal_evidence and bool(requested_standards)

        try:
            rows = await self.tools.retrieve_summaries(
                query=query,
                tenant_id=tenant_id,
                k=plan.summary_k,
                collection_id=collection_id,
            )
        except Exception:
            rows = []

        if self.allowed_source_ids:
            filtered: list[dict[str, Any]] = []
            for row in rows:
                meta = row.get("metadata") or {}
                row_source = str(row.get("source_id") or row.get("id") or "")
                row_collection_id = str(meta.get("collection_id") or "")
                row_collection_name = str(meta.get("collection_name") or "")
                if row_source in self.allowed_source_ids:
                    filtered.append(row)
                    continue
                if collection_id and row_collection_id == str(collection_id):
                    filtered.append(row)
                    continue
                if (
                    self.collection_name
                    and row_collection_name.lower() == str(self.collection_name).lower()
                ):
                    filtered.append(row)
            rows = filtered

        filtered_by_scope = [r for r in rows if _row_matches_standards(r, requested_standards)]
        if strict_scope and bool(getattr(settings, "SCOPE_STRICT_FILTERING", False)):
            rows = filtered_by_scope
        elif strict_scope and filtered_by_scope:
            rows = filtered_by_scope
        elif not strict_scope:
            rows = filtered_by_scope or rows

        if plan.require_literal_evidence:
            rows = _rerank_for_literal(query, rows)

        return [
            EvidenceItem(
                source=str(row.get("id") or f"summary-{i}"),
                content=str(row.get("content") or "").strip(),
                score=float(row.get("similarity") or 0.0),
                metadata={"row": row},
            )
            for i, row in enumerate(rows)
            if row.get("content")
        ]


@dataclass
class GroqAnswerGeneratorAdapter:
    client: Any
    model_name: str

    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
        agent_profile: AgentProfile | None = None,
    ) -> AnswerDraft:
        if not chunks and not summaries:
            return AnswerDraft(
                text="⚠️ No pude encontrar información relevante en el contexto recuperado.",
                mode=plan.mode,
                evidence=[],
            )

        all_evidence = [*chunks, *summaries]
        evidence_block = "\n\n".join(f"[{item.source}] {item.content}" for item in all_evidence)
        context = f"=== FRAGMENTOS DE EVIDENCIA ===\n{evidence_block}"

        prompt = f"""
Eres un experto respondiendo consultas con precisión forense.
Basarás tu respuesta ESTRICTAMENTE en los fragmentos de texto provistos.
No uses conocimiento externo.
Cada afirmación clave debe terminar con la cita del ID del fragmento de origen entre corchetes ASCII estándar, ejemplo: [b61b3913-16be-4e52-9e82-cc82918b6b25].
IMPORTANTE: Usa SOLO corchetes estándar [ ] y NO uses corchetes especiales como 【 】.
Si los fragmentos no contienen la respuesta, di explícitamente que no hay información suficiente.

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            messages=[{"role": "user", "content": prompt.strip()}],
            model=self.model_name,
            temperature=0.05,
        )
        text = str(completion.choices[0].message.content or "").strip()
        if not text:
            if chunks:
                bullets = []
                for item in chunks[:3]:
                    snippet = (item.content or "").replace("\n", " ").strip()
                    if len(snippet) > 220:
                        snippet = snippet[:220].rstrip() + "..."
                    bullets.append(f"- {snippet} Fuente({item.source})")
                text = (
                    "No hubo salida textual del modelo. Resumen mínimo desde evidencia recuperada:\n"
                    + "\n".join(bullets)
                )
            else:
                text = (
                    "No encontrado explicitamente en el contexto recuperado. "
                    f"No puedo emitir una conclusion confiable sin evidencia trazable adicional ({citation_format})."
                )
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])


class LiteralEvidenceValidator:
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str) -> ValidationResult:
        blocking_issues: list[str] = []
        warnings: list[str] = []

        def _add_issue(message: str) -> None:
            blocking_issues.append(message)

        def _add_scope_guardrail(message: str) -> None:
            # Scope guardrails are strict only in literal evidence mode.
            if plan.require_literal_evidence:
                blocking_issues.append(message)
                return
            warnings.append(f"Warning: {message}")

        enforce_clause_refs = bool(plan.require_literal_evidence)

        def _extract_section_block(text: str, section_name: str) -> str:
            pattern = re.compile(
                rf"(?:^|\n)\s*(?:##\s*)?{re.escape(section_name)}\s*:?\s*\n(?P<body>.*?)(?=\n\s*(?:##\s*)?[A-Za-zÁÉÍÓÚáéíóúñÑ].*?:\s*\n|\Z)",
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(text or "")
            return str(match.group("body") if match else "").strip()

        def _extract_citation_ids(text: str) -> set[str]:
            # Extracción estricta formato [chunk_id] — acepta [] y full-width 【】
            matches = re.findall(r"[\[\uff3b\u3010]([a-f0-9\-]{8,})[\]\uff3d\u3011]", text or "", re.IGNORECASE)
            return {m.strip() for m in matches if m.strip()}

        if plan.require_literal_evidence and not draft.evidence:
            _add_issue("No retrieval evidence available for literal answer mode.")

        # GUILLOTINA DE CITAS
        cited_ids = _extract_citation_ids(draft.text)
        evidence_ids = {str(ev.source).strip() for ev in draft.evidence}
        
        fallback_detected = re.search(r"no hay (?:evidencia|informaci[oó]n)", str(draft.text or "").lower())
        
        if not fallback_detected:
            if not cited_ids and draft.evidence:
                _add_issue("Answer does not include explicit source markers in [chunk_id] format.")
                
            hallucinated_ids = cited_ids - evidence_ids
            if hallucinated_ids:
                _add_issue(f"Hallucinated citations detected: {', '.join(hallucinated_ids)} are not valid evidence IDs.")

        requested = plan.requested_standards or extract_requested_scopes(query)
        mentioned_in_answer = {item.upper() for item in extract_requested_scopes(draft.text)}
        requested_upper = {item.upper() for item in requested}

        if (
            requested_upper
            and mentioned_in_answer
            and not mentioned_in_answer.issubset(requested_upper)
        ):
            _add_scope_guardrail(
                "Scope mismatch detected: answer mentions a different standard than the query scope."
            )

        if requested and draft.evidence:
            mismatched = 0
            total_with_scope = 0
            covered_requested: set[str] = set()
            query_clause_refs = sorted(_extract_clause_refs(query)) if enforce_clause_refs else []
            matched_clause_refs: set[str] = set()
            for ev in draft.evidence:
                row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else {}
                if not isinstance(row, dict):
                    continue
                row_scope = _extract_row_standard(row)
                if not row_scope:
                    continue
                total_with_scope += 1
                matched_scope = False
                for target in requested_upper:
                    if _scope_matches(row_scope, target):
                        covered_requested.add(target)
                        matched_scope = True
                if not matched_scope:
                    mismatched += 1

                if query_clause_refs:
                    content = str(row.get("content") or "")
                    meta_refs = _extract_metadata_clause_refs(row)
                    for ref in query_clause_refs:
                        in_content = bool(re.search(rf"\b{re.escape(ref)}(?:\.\d+)*\b", content))
                        in_meta = any(_clause_ref_matches(ref, meta_ref) for meta_ref in meta_refs)
                        if in_content or in_meta:
                            matched_clause_refs.add(ref)

            if total_with_scope > 0 and mismatched > 0:
                _add_scope_guardrail(
                    "Scope mismatch detected: evidence includes sources outside requested standard scope."
                )

            if len(requested_upper) >= 2:
                missing_scope_coverage = sorted(
                    scope for scope in requested_upper if scope not in covered_requested
                )
                if missing_scope_coverage:
                    _add_scope_guardrail(
                        "Scope mismatch detected: missing evidence coverage for requested standards: "
                        + ", ".join(missing_scope_coverage)
                    )

            if query_clause_refs:
                requested_count = len(query_clause_refs)
                matched_count = len(matched_clause_refs)
                semantic_hits = 0
                if matched_count == 0:
                    for ev in draft.evidence:
                        row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else {}
                        if not isinstance(row, dict):
                            continue
                        if _semantic_clause_match(
                            query=query, row=row, requested_upper=requested_upper
                        ):
                            semantic_hits += 1
                if requested_count <= 2:
                    required_matches = requested_count
                else:
                    required_matches = max(
                        2,
                        int(
                            math.ceil(
                                requested_count
                                * float(
                                    getattr(settings, "ORCH_LITERAL_REF_MIN_COVERAGE_RATIO", 0.7)
                                )
                            )
                        ),
                    )

                # Operational policy: in literal modes allow grounded partial inference
                # when semantic evidence exists, instead of blocking with null answer.
                if matched_count < required_matches and semantic_hits == 0:
                    coverage_ratio = (
                        round(matched_count / requested_count, 3) if requested_count else 0.0
                    )
                    _add_issue(
                        "Literal clause coverage insufficient: "
                        f"matched {matched_count}/{requested_count} refs "
                        f"(required {required_matches}, ratio={coverage_ratio})."
                    )

            if query_clause_refs and not matched_clause_refs:
                semantic_hits = 0
                for ev in draft.evidence:
                    row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else {}
                    if not isinstance(row, dict):
                        continue
                    if _semantic_clause_match(
                        query=query, row=row, requested_upper=requested_upper
                    ):
                        semantic_hits += 1

                if semantic_hits == 0:
                    _add_issue(
                        "Literal clause mismatch: no evidence chunk contains the requested clause reference."
                    )

        return ValidationResult(
            accepted=not blocking_issues,
            issues=[*blocking_issues, *warnings],
        )
