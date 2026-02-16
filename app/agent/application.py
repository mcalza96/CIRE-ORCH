from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

import structlog

from app.agent.models import (
    AnswerDraft,
    ClarificationRequest,
    EvidenceItem,
    QueryIntent,
    QueryMode,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.cartridges.models import AgentProfile
from app.agent.errors import ScopeValidationError
from app.agent.retrieval_planner import build_initial_scope_filters
from app.agent.mode_advisor import ModeAdvisor
from app.core.config import settings


class RetrieverPort(Protocol):
    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]: ...

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]: ...

    # Optional methods supported by advanced contract adapters.
    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def apply_validated_scope(self, validated: dict[str, Any]) -> None: ...


class AnswerGeneratorPort(Protocol):
    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
        agent_profile: AgentProfile | None = None,
    ) -> AnswerDraft: ...


class ValidatorPort(Protocol):
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str) -> ValidationResult: ...


from app.agent.policies import (
    build_retrieval_plan,
    classify_intent,
    classify_intent_with_trace,
    detect_conflict_objectives,
    detect_scope_candidates,
    extract_requested_scopes,
    has_clause_reference,
    suggest_scope_candidates,
)


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class HandleQuestionCommand:
    query: str
    tenant_id: str
    collection_id: str | None
    scope_label: str
    user_id: str | None = None
    agent_profile: AgentProfile | None = None


@dataclass(frozen=True)
class HandleQuestionResult:
    intent: QueryIntent
    plan: RetrievalPlan
    answer: AnswerDraft
    validation: ValidationResult
    retrieval: RetrievalDiagnostics
    clarification: ClarificationRequest | None = None


class HandleQuestionUseCase:
    """Application orchestrator for Q/A Orchestrator."""

    def __init__(
        self,
        retriever: RetrieverPort,
        answer_generator: AnswerGeneratorPort,
        validator: ValidatorPort,
    ):
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._validator = validator

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        def _parse_explicit_mode_override(text: str) -> QueryMode | None:
            raw = (text or "").lower()
            m = re.search(r"__mode__\s*=\s*(\w+)", raw)
            if not m:
                m = re.search(r"__clarified_mode__\s*=\s*(\w+)", raw)
            if not m:
                return None
            value = m.group(1).strip()
            mapping = {
                "comparativa": "comparativa",
                "explicativa": "explicativa",
                "literal_normativa": "literal_normativa",
                "literal_lista": "literal_lista",
                "ambigua_scope": "ambigua_scope",
            }
            return cast(QueryMode | None, mapping.get(value))

        def _extract_row_standard(row: dict[str, Any]) -> str:
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            candidates = [
                meta.get("source_standard"),
                meta.get("standard"),
                meta.get("scope"),
                row.get("source_standard"),
            ]
            for value in candidates:
                if isinstance(value, str) and value.strip():
                    return value.strip().upper()
            return ""

        def _filter_evidence_by_standards(
            evidence: list[EvidenceItem],
            *,
            allowed_standards: tuple[str, ...],
        ) -> list[EvidenceItem]:
            if not allowed_standards:
                return evidence
            allowed_upper = {s.upper() for s in allowed_standards if s}
            out: list[EvidenceItem] = []
            for item in evidence:
                row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
                if not isinstance(row, dict):
                    out.append(item)
                    continue
                std = _extract_row_standard(row)
                if not std:
                    out.append(item)
                    continue
                if any(target in std for target in allowed_upper):
                    out.append(item)
            return out

        normalized_query = (cmd.query or "").lower()
        has_user_clarification = (
            "aclaración de alcance:" in normalized_query
            or "aclaracion de alcance:" in normalized_query
            or "modo preferido en sesión:" in normalized_query
            or "modo preferido en sesion:" in normalized_query
            or "__clarified_scope__=true" in normalized_query
        )

        intent_trace: dict[str, Any] = {}
        explicit_mode = _parse_explicit_mode_override(cmd.query)
        mode_clarified = explicit_mode is not None
        intent = classify_intent(cmd.query, profile=cmd.agent_profile)
        if settings.ORCH_MODE_CLASSIFIER_V2:
            intent, intent_trace = classify_intent_with_trace(cmd.query, profile=cmd.agent_profile)

        if explicit_mode is not None and explicit_mode != intent.mode:
            intent = QueryIntent(mode=explicit_mode, rationale="explicit_mode_override")
            if isinstance(intent_trace, dict):
                intent_trace["explicit_override"] = True
                intent_trace["explicit_mode"] = explicit_mode

        # Optional LLM advisor for low-confidence classifications.
        if (
            settings.ORCH_MODE_ADVISOR_ENABLED
            and isinstance(intent_trace, dict)
            and float(intent_trace.get("confidence") or 1.0)
            < float(settings.ORCH_MODE_LOW_CONFIDENCE_THRESHOLD or 0.55)
        ):
            blocked = set(
                item for item in (intent_trace.get("blocked_modes") or []) if isinstance(item, str)
            )
            candidates = cast(
                tuple[QueryMode, ...],
                tuple(
                    m
                    for m in (
                        "comparativa",
                        "explicativa",
                        "literal_normativa",
                        "literal_lista",
                        "ambigua_scope",
                    )
                    if m not in blocked
                ),
            )
            suggestion = await ModeAdvisor().suggest(
                query=cmd.query,
                candidate_modes=candidates,
                profile=cmd.agent_profile,
            )
            if suggestion is not None and suggestion.mode not in blocked:
                intent_trace["advisor_used"] = True
                intent_trace["advisor_suggestion"] = {
                    "mode": suggestion.mode,
                    "confidence": suggestion.confidence,
                    "rationale": suggestion.rationale,
                }
                # Kernel chooses the advisor mode only when it differs and is not blocked.
                if suggestion.mode != intent.mode:
                    intent = QueryIntent(
                        mode=suggestion.mode,
                        rationale=f"advisor override: {suggestion.rationale}".strip(),
                    )

        plan = build_retrieval_plan(intent, query=cmd.query, profile=cmd.agent_profile)
        detected_scopes = detect_scope_candidates(cmd.query, profile=cmd.agent_profile)
        conflict_mode = detect_conflict_objectives(cmd.query, profile=cmd.agent_profile)

        def _build_profile_clarification() -> ClarificationRequest | None:
            profile = cmd.agent_profile
            if profile is None or has_user_clarification:
                return None
            scopes_text = ", ".join(detected_scopes)
            for rule in profile.clarification_rules:
                if not isinstance(rule, dict):
                    continue
                # 1. Scope mismatch trigger (min_scope_count)
                min_scope_raw: Any = rule.get("min_scope_count")
                if min_scope_raw is not None:
                    try:
                        min_scope_count = int(min_scope_raw)
                        if len(detected_scopes) < min_scope_count:
                            continue
                    except (ValueError, TypeError):
                        pass

                # 2. Mode trigger
                target_mode = rule.get("mode")
                if target_mode and target_mode != intent.mode:
                    continue

                # 3. Marker triggers
                all_markers = [str(m).lower() for m in rule.get("all_markers", [])]
                any_markers = [str(m).lower() for m in rule.get("any_markers", [])]

                virtual_markers = {f"__mode__={intent.mode}"}
                confidence = float(intent_trace.get("confidence") or 1.0)
                if confidence < float(settings.ORCH_MODE_LOW_CONFIDENCE_THRESHOLD or 0.55):
                    virtual_markers.add("__low_confidence__")

                def _match_marker(m: str) -> bool:
                    return m in normalized_query or m in virtual_markers

                if all_markers and not all(_match_marker(m) for m in all_markers):
                    continue
                if any_markers and not any(_match_marker(m) for m in any_markers):
                    continue

                template = str(rule.get("question_template") or rule.get("question") or "").strip()
                if not template:
                    continue
                question = template.format(scopes=scopes_text)
                options = tuple(str(o).strip() for o in rule.get("options", []))
                return ClarificationRequest(question=question, options=options)
            return None

        if conflict_mode and has_user_clarification:
            plan = RetrievalPlan(
                mode="explicativa",
                chunk_k=max(plan.chunk_k, 35),
                chunk_fetch_k=max(plan.chunk_fetch_k, 140),
                summary_k=max(plan.summary_k, 4),
                require_literal_evidence=False,
                requested_standards=plan.requested_standards,
            )

        clarification_from_profile = _build_profile_clarification()
        if clarification_from_profile is not None:
            clarification = clarification_from_profile
            answer = AnswerDraft(text=clarification.question, mode=plan.mode, evidence=[])
            validation = ValidationResult(accepted=True, issues=[])
            return HandleQuestionResult(
                intent=intent,
                plan=plan,
                answer=answer,
                validation=validation,
                retrieval=RetrievalDiagnostics(
                    contract="legacy",
                    strategy="clarification",
                    partial=False,
                    trace={},
                    scope_validation={},
                ),
                clarification=clarification,
            )

        # Advanced retrieval contract: validate scope first to avoid executing retrieval with invalid filters
        # and to surface scope clarification hints early.
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced" and hasattr(
            self._retriever, "validate_scope"
        ):
            initial_filters = build_initial_scope_filters(
                plan_requested=plan.requested_standards,
                mode=plan.mode,
                query=cmd.query,
                profile=cmd.agent_profile,
            )
            validated = await self._retriever.validate_scope(
                query=cmd.query,
                tenant_id=cmd.tenant_id,
                collection_id=cmd.collection_id,
                plan=plan,
                user_id=cmd.user_id,
                filters=initial_filters,
            )
            valid = bool(validated.get("valid", True)) if isinstance(validated, dict) else True
            if not valid:
                violations = validated.get("violations") if isinstance(validated, dict) else None
                warnings = validated.get("warnings") if isinstance(validated, dict) else None
                normalized = (
                    validated.get("normalized_scope") if isinstance(validated, dict) else None
                )
                query_scope = validated.get("query_scope") if isinstance(validated, dict) else None
                raise ScopeValidationError(
                    message="Scope validation failed",
                    violations=list(violations) if isinstance(violations, list) else [],
                    warnings=list(warnings) if isinstance(warnings, list) else None,
                    normalized_scope=normalized if isinstance(normalized, dict) else None,
                    query_scope=query_scope if isinstance(query_scope, dict) else None,
                )

            query_scope = validated.get("query_scope") if isinstance(validated, dict) else {}
            requires_scope_clarification = (
                bool(query_scope.get("requires_scope_clarification", False))
                if isinstance(query_scope, dict)
                else False
            )
            suggested_scopes = (
                query_scope.get("suggested_scopes") if isinstance(query_scope, dict) else []
            )
            if requires_scope_clarification and not has_user_clarification:
                options = tuple(
                    str(item)
                    for item in (suggested_scopes if isinstance(suggested_scopes, list) else [])
                    if str(item).strip()
                )
                clarification = ClarificationRequest(
                    question=(
                        "Detecté ambigüedad de alcance. ¿Quieres análisis integrado o restringirlo a una norma?"
                    ),
                    options=options,
                )
                answer = AnswerDraft(text=clarification.question, mode=plan.mode, evidence=[])
                validation = ValidationResult(accepted=True, issues=[])
                return HandleQuestionResult(
                    intent=intent,
                    plan=plan,
                    answer=answer,
                    validation=validation,
                    retrieval=RetrievalDiagnostics(
                        contract="advanced",
                        strategy="validate_scope",
                        partial=False,
                        trace={},
                        scope_validation=validated if isinstance(validated, dict) else {},
                    ),
                    clarification=clarification,
                )

            if hasattr(self._retriever, "apply_validated_scope") and isinstance(validated, dict):
                try:
                    self._retriever.apply_validated_scope(validated)
                except Exception:
                    pass

        # Retrieval execution is delegated to the RetrieverPort implementation.
        # Legacy retrievers may return chunks + summaries; advanced contract retrievers can
        # return all evidence as "chunks" and keep "summaries" empty.
        async def _run_retrieval(
            *, attempt_plan: RetrievalPlan
        ) -> tuple[list[EvidenceItem], list[EvidenceItem], RetrievalDiagnostics]:
            # If we are in advanced contract mode and the plan changed (e.g. mode auto-retry),
            # re-validate scope to avoid stale clause_id filters.
            if (
                str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced"
                and hasattr(self._retriever, "validate_scope")
                and hasattr(self._retriever, "apply_validated_scope")
            ):
                try:
                    attempt_filters = build_initial_scope_filters(
                        plan_requested=attempt_plan.requested_standards,
                        mode=attempt_plan.mode,
                        query=cmd.query,
                        profile=cmd.agent_profile,
                    )
                    validated_attempt = await self._retriever.validate_scope(
                        query=cmd.query,
                        tenant_id=cmd.tenant_id,
                        collection_id=cmd.collection_id,
                        plan=attempt_plan,
                        user_id=cmd.user_id,
                        filters=attempt_filters,
                    )
                    if isinstance(validated_attempt, dict):
                        self._retriever.apply_validated_scope(validated_attempt)
                except Exception:
                    pass

            chunks_task = asyncio.create_task(
                self._retriever.retrieve_chunks(
                    query=cmd.query,
                    tenant_id=cmd.tenant_id,
                    collection_id=cmd.collection_id,
                    plan=attempt_plan,
                    user_id=cmd.user_id,
                )
            )
            summaries_task = asyncio.create_task(
                self._retriever.retrieve_summaries(
                    query=cmd.query,
                    tenant_id=cmd.tenant_id,
                    collection_id=cmd.collection_id,
                    plan=attempt_plan,
                    user_id=cmd.user_id,
                )
            )
            chunks_result, summaries_result = await asyncio.gather(
                chunks_task, summaries_task, return_exceptions=True
            )

            chunks: list[EvidenceItem] = []
            summaries: list[EvidenceItem] = []
            retrieval = RetrievalDiagnostics(
                contract="legacy",
                strategy="legacy",
                partial=False,
                trace={},
                scope_validation={},
            )

            chunks_error = chunks_result if isinstance(chunks_result, BaseException) else None
            summaries_error = (
                summaries_result if isinstance(summaries_result, BaseException) else None
            )

            if chunks_error is None and isinstance(chunks_result, list):
                chunks = cast(list[EvidenceItem], chunks_result)
            if summaries_error is None and isinstance(summaries_result, list):
                summaries = cast(list[EvidenceItem], summaries_result)

            # Advanced contract retrievers can attach retrieval diagnostics to the instance.
            diag = getattr(self._retriever, "last_retrieval_diagnostics", None)
            if isinstance(diag, RetrievalDiagnostics):
                retrieval = diag

            if chunks_error and summaries_error:
                logger.error(
                    "rag_retrieval_failed",
                    chunks_error=str(chunks_error),
                    summaries_error=str(summaries_error),
                )
                raise RuntimeError(
                    "RAG retrieval failed for both chunks and summaries"
                ) from chunks_error

            if chunks_error or summaries_error:
                logger.warning(
                    "rag_retrieval_degraded",
                    chunks_failed=chunks_error is not None,
                    summaries_failed=summaries_error is not None,
                    chunks_error=str(chunks_error) if chunks_error else None,
                    summaries_error=str(summaries_error) if summaries_error else None,
                )

            return chunks, summaries, retrieval

        # Level 4-ish internal correction loop: a small, bounded retry with working memory.
        attempts_trace: list[dict[str, Any]] = []

        current_plan = plan
        chunks, summaries, retrieval = await _run_retrieval(attempt_plan=current_plan)

        initial_plan = current_plan
        initial_chunks = chunks
        initial_summaries = summaries
        initial_retrieval = retrieval

        # Auto-retry on retrieval collapse when the initial mode was overly strict.
        if (
            settings.ORCH_MODE_AUTORETRY_ENABLED
            and int(settings.ORCH_MODE_AUTORETRY_MAX_ATTEMPTS or 2) >= 2
            and not chunks
            and not summaries
            and current_plan.mode in {"literal_normativa", "literal_lista"}
        ):
            fallback_mode: QueryMode = (
                "comparativa" if len(current_plan.requested_standards) >= 2 else "explicativa"
            )
            if fallback_mode != current_plan.mode:
                retry_intent = QueryIntent(
                    mode=fallback_mode,
                    rationale="auto_retry_on_empty_retrieval",
                )
                retry_plan = build_retrieval_plan(
                    retry_intent,
                    query=cmd.query,
                    profile=cmd.agent_profile,
                )
                chunks2, summaries2, retrieval2 = await _run_retrieval(attempt_plan=retry_plan)
                current_plan = retry_plan
                chunks, summaries, retrieval = chunks2, summaries2, retrieval2

        # HITL fallback for low-confidence queries that still produced no evidence.
        if (
            settings.ORCH_MODE_HITL_ENABLED
            and not mode_clarified
            and not chunks
            and not summaries
            and isinstance(intent_trace, dict)
            and float(intent_trace.get("confidence") or 1.0)
            < float(settings.ORCH_MODE_LOW_CONFIDENCE_THRESHOLD or 0.55)
        ):
            clarification = ClarificationRequest(
                question=(
                    "No pude recuperar evidencia suficiente y la pregunta es ambigua en cuanto al tipo de respuesta. "
                    "Elige el modo: comparativa (integrada), explicativa (analitica) o literal_normativa (citas exactas)."
                ),
                options=("comparativa", "explicativa", "literal_normativa"),
            )
            answer = AnswerDraft(text=clarification.question, mode=current_plan.mode, evidence=[])
            validation = ValidationResult(accepted=True, issues=[])
            trace_payload = dict(retrieval.trace or {})
            trace_payload["mode_clarification"] = {
                "intent": intent.mode,
                "confidence": float(intent_trace.get("confidence") or 0.0),
            }
            return HandleQuestionResult(
                intent=intent,
                plan=current_plan,
                answer=answer,
                validation=validation,
                retrieval=RetrievalDiagnostics(
                    contract=retrieval.contract,
                    strategy="clarification_mode",
                    partial=bool(retrieval.partial),
                    trace=trace_payload,
                    scope_validation=retrieval.scope_validation,
                ),
                clarification=clarification,
            )
        # --------------------------------------------------------------------------
        # PHASE 2: Profile-driven Clarifications (Declarative Engine)
        # --------------------------------------------------------------------------
        # Profile rules handled higher up in Execute if applicable before retrieval.
        # This section is for post-retrieval corrections if needed.
        
        # Legacy fallback/cleanup for structural ambiguity
        if (
            not has_user_clarification
            and intent.mode == "explicativa"
            and len(detected_scopes) >= 2
            and len(plan.requested_standards) < 2
            and has_clause_reference(cmd.query, profile=cmd.agent_profile)
        ):
            # ... keep existing structural ambiguity logic if needed ...
            pass
        answer = await self._answer_generator.generate(
            query=cmd.query,
            scope_label=cmd.scope_label,
            plan=current_plan,
            chunks=chunks,
            summaries=summaries,
            agent_profile=cmd.agent_profile,
        )
        validation = self._validator.validate(answer, current_plan, cmd.query)

        attempts_trace.append(
            {
                "attempt": 1,
                "plan": {
                    "mode": initial_plan.mode,
                    "chunk_k": initial_plan.chunk_k,
                    "chunk_fetch_k": initial_plan.chunk_fetch_k,
                    "summary_k": initial_plan.summary_k,
                    "requested_standards": list(initial_plan.requested_standards),
                },
                "retrieval": {
                    "contract": initial_retrieval.contract,
                    "strategy": initial_retrieval.strategy,
                    "partial": bool(initial_retrieval.partial),
                },
                "validation": {
                    "accepted": bool(validation.accepted)
                    if (initial_chunks or initial_summaries)
                    else False,
                    "issues": list(validation.issues)
                    if (initial_chunks or initial_summaries)
                    else ["retrieval_empty"],
                },
                "action": "initial",
            }
        )

        if initial_plan.mode != current_plan.mode:
            attempts_trace.append(
                {
                    "attempt": 2,
                    "plan": {
                        "mode": current_plan.mode,
                        "chunk_k": current_plan.chunk_k,
                        "chunk_fetch_k": current_plan.chunk_fetch_k,
                        "summary_k": current_plan.summary_k,
                        "requested_standards": list(current_plan.requested_standards),
                    },
                    "retrieval": {
                        "contract": retrieval.contract,
                        "strategy": retrieval.strategy,
                        "partial": bool(retrieval.partial),
                    },
                    "validation": {
                        "accepted": bool(validation.accepted),
                        "issues": list(validation.issues),
                    },
                    "action": "auto_retry_mode",
                    "retry_from": initial_plan.mode,
                    "retry_to": current_plan.mode,
                }
            )

        if not validation.accepted:
            issues = list(validation.issues)
            issue_text = " | ".join(issues)
            action = ""

            # Prefer silent correction before asking the user.
            scope_answer_mismatch = any("answer mentions" in it.lower() for it in issues)
            scope_evidence_mismatch = any("evidence includes" in it.lower() for it in issues)
            clause_mismatch = any("literal clause mismatch" in it.lower() for it in issues)
            missing_citations = any("explicit source markers" in it.lower() for it in issues)
            no_evidence = any("no retrieval evidence" in it.lower() for it in issues)

            # Attempt 2 is bounded and only runs when it can plausibly improve.
            if scope_answer_mismatch or scope_evidence_mismatch:
                # Tighten synthesis scope: only allow evidence within requested standards.
                filtered_evidence = _filter_evidence_by_standards(
                    [*chunks, *summaries],
                    allowed_standards=current_plan.requested_standards,
                )
                # Split back into chunks/summaries preserving original labels.
                chunks2 = [it for it in filtered_evidence if str(it.source or "").startswith("C")]
                summaries2 = [
                    it for it in filtered_evidence if str(it.source or "").startswith("R")
                ]
                if filtered_evidence and (chunks2 or summaries2):
                    action = "filter_evidence_to_scope_and_regenerate"
                    answer2 = await self._answer_generator.generate(
                        query=cmd.query
                        + "\n\n[INSTRUCCION INTERNA] No menciones normas fuera del alcance solicitado. "
                        + (
                            "Alcance: " + ", ".join(current_plan.requested_standards)
                            if current_plan.requested_standards
                            else ""
                        ),
                        scope_label=cmd.scope_label,
                        plan=current_plan,
                        chunks=chunks2,
                        summaries=summaries2,
                        agent_profile=cmd.agent_profile,
                    )
                    validation2 = self._validator.validate(answer2, current_plan, cmd.query)
                    if validation2.accepted:
                        answer = answer2
                        validation = validation2
                        chunks = chunks2
                        summaries = summaries2
                    attempts_trace.append(
                        {
                            "attempt": 2,
                            "plan": {
                                "mode": current_plan.mode,
                                "chunk_k": current_plan.chunk_k,
                                "chunk_fetch_k": current_plan.chunk_fetch_k,
                                "summary_k": current_plan.summary_k,
                                "requested_standards": list(current_plan.requested_standards),
                            },
                            "retrieval": {
                                "contract": retrieval.contract,
                                "strategy": retrieval.strategy,
                                "partial": bool(retrieval.partial),
                            },
                            "validation": {
                                "accepted": bool(validation2.accepted),
                                "issues": list(validation2.issues),
                            },
                            "action": action,
                        }
                    )
            elif clause_mismatch or no_evidence:
                # Increase recall for literal clause questions.
                action = "increase_fetch_k_and_reretrieve"
                boosted = RetrievalPlan(
                    mode=current_plan.mode,
                    chunk_k=max(current_plan.chunk_k, 55),
                    chunk_fetch_k=max(current_plan.chunk_fetch_k, 280),
                    summary_k=current_plan.summary_k,
                    require_literal_evidence=current_plan.require_literal_evidence,
                    requested_standards=current_plan.requested_standards,
                )
                chunks2, summaries2, retrieval2 = await _run_retrieval(attempt_plan=boosted)
                answer2 = await self._answer_generator.generate(
                    query=cmd.query,
                    scope_label=cmd.scope_label,
                    plan=boosted,
                    chunks=chunks2,
                    summaries=summaries2,
                    agent_profile=cmd.agent_profile,
                )
                validation2 = self._validator.validate(answer2, boosted, cmd.query)
                attempts_trace.append(
                    {
                        "attempt": 2,
                        "plan": {
                            "mode": boosted.mode,
                            "chunk_k": boosted.chunk_k,
                            "chunk_fetch_k": boosted.chunk_fetch_k,
                            "summary_k": boosted.summary_k,
                            "requested_standards": list(boosted.requested_standards),
                        },
                        "retrieval": {
                            "contract": retrieval2.contract,
                            "strategy": retrieval2.strategy,
                            "partial": bool(retrieval2.partial),
                        },
                        "validation": {
                            "accepted": bool(validation2.accepted),
                            "issues": list(validation2.issues),
                        },
                        "action": action,
                    }
                )
                if validation2.accepted:
                    current_plan = boosted
                    chunks, summaries, retrieval = chunks2, summaries2, retrieval2
                    answer, validation = answer2, validation2
            elif missing_citations:
                # Try a stricter re-generation without changing retrieval.
                action = "regenerate_with_strict_citation_instruction"
                answer2 = await self._answer_generator.generate(
                    query=cmd.query
                    + "\n\n[INSTRUCCION INTERNA] Incluye marcadores C#/R# en cada afirmacion clave.",
                    scope_label=cmd.scope_label,
                    plan=current_plan,
                    chunks=chunks,
                    summaries=summaries,
                    agent_profile=cmd.agent_profile,
                )
                validation2 = self._validator.validate(answer2, current_plan, cmd.query)
                attempts_trace.append(
                    {
                        "attempt": 2,
                        "plan": {
                            "mode": current_plan.mode,
                            "chunk_k": current_plan.chunk_k,
                            "chunk_fetch_k": current_plan.chunk_fetch_k,
                            "summary_k": current_plan.summary_k,
                            "requested_standards": list(current_plan.requested_standards),
                        },
                        "retrieval": {
                            "contract": retrieval.contract,
                            "strategy": retrieval.strategy,
                            "partial": bool(retrieval.partial),
                        },
                        "validation": {
                            "accepted": bool(validation2.accepted),
                            "issues": list(validation2.issues),
                        },
                        "action": action,
                    }
                )
                if validation2.accepted:
                    answer, validation = answer2, validation2
            else:
                logger.info(
                    "orchestrator_no_silent_retry",
                    tenant_id=cmd.tenant_id,
                    mode=current_plan.mode,
                    issues=issue_text,
                )
        if not validation.accepted and any(
            "Scope mismatch" in issue for issue in validation.issues
        ):
            if len(detected_scopes) >= 2:
                clarification = ClarificationRequest(
                    question=(
                        "La consulta parece cruzar multiples alcances ("
                        + ", ".join(detected_scopes)
                        + "). ¿Confirmas analisis integrado o prefieres restringir el alcance?"
                    ),
                    options=("Analisis integrado", *detected_scopes),
                )
                answer = AnswerDraft(
                    text=clarification.question, mode=plan.mode, evidence=answer.evidence
                )
                validation = ValidationResult(accepted=True, issues=[])
                return HandleQuestionResult(
                    intent=intent,
                    plan=plan,
                    answer=answer,
                    validation=validation,
                    retrieval=retrieval,
                    clarification=clarification,
                )

            answer = AnswerDraft(
                text=(
                    "⚠️ Respuesta bloqueada por inconsistencia de ámbito entre la pregunta y las fuentes recuperadas. "
                    "Reformula indicando explícitamente el alcance objetivo."
                ),
                mode=plan.mode,
                evidence=answer.evidence,
            )

        # Attach attempt history to retrieval trace for white-box debugging.
        trace_out = dict(retrieval.trace or {})
        if isinstance(intent_trace, dict) and intent_trace:
            # Keep payload bounded.
            trace_out["classification"] = {
                "version": intent_trace.get("version"),
                "mode": intent_trace.get("mode") or intent.mode,
                "confidence": intent_trace.get("confidence"),
                "reasons": intent_trace.get("reasons"),
                "blocked_modes": intent_trace.get("blocked_modes"),
                "advisor_used": bool(intent_trace.get("advisor_used", False)),
                "advisor_suggestion": intent_trace.get("advisor_suggestion"),
                "explicit_override": bool(intent_trace.get("explicit_override", False)),
                "explicit_mode": intent_trace.get("explicit_mode"),
                "features": intent_trace.get("features"),
            }
        if cmd.agent_profile is not None:
            trace_out["agent_profile"] = {
                "profile_id": cmd.agent_profile.profile_id,
                "version": cmd.agent_profile.version,
                "status": cmd.agent_profile.status,
            }
        if attempts_trace:
            trace_out["attempts"] = attempts_trace
        retrieval = RetrievalDiagnostics(
            contract=retrieval.contract,
            strategy=retrieval.strategy,
            partial=bool(retrieval.partial),
            trace=trace_out,
            scope_validation=dict(retrieval.scope_validation or {}),
        )

        return HandleQuestionResult(
            intent=intent,
            plan=current_plan,
            answer=answer,
            validation=validation,
            retrieval=retrieval,
        )
