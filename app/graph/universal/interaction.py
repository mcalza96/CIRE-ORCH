from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agent.models import QueryIntent, ReasoningPlan, RetrievalPlan
from app.agent.policies import detect_scope_candidates
from app.cartridges.models import AgentProfile, ModeInteractionPolicy


@dataclass(frozen=True)
class InteractionDecision:
    level: str
    needs_interrupt: bool
    kind: str
    question: str
    options: tuple[str, ...]
    metrics: dict[str, Any]
    missing_slots: tuple[str, ...] = ()
    scope_candidates: tuple[str, ...] = ()


def _query_has_user_feedback(query: str) -> bool:
    lowered = str(query or "").lower()
    markers = (
        "__plan_approved__=true",
        "__plan_feedback__=",
    )
    return any(marker in lowered for marker in markers)


def _clarification_round(query: str) -> int:
    match = re.search(r"__clarification_round__=(\d+)", str(query or ""), flags=re.IGNORECASE)
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _clarification_choice(query: str) -> str:
    match = re.search(
        r"__clarification_choice__=([a-z0-9_\-]{2,64})",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return str(match.group(1) if match else "").strip().lower()


def _clarification_confirmed(query: str) -> bool:
    return bool(
        re.search(r"__clarification_confirmed__=true", str(query or ""), flags=re.IGNORECASE)
    )


def _clarification_text(query: str) -> str:
    match = re.search(
        r"Aclaraci[oó]n\s+de\s+alcance:\s*(.+)$",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _looks_like_scope_phrase(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if re.search(
        r"\b(?:iso|iec|nom|nmx|nfpa|osha|en|une|iram|bs|din)\b", value, flags=re.IGNORECASE
    ):
        return True
    return bool(re.search(r"\d", value))


def _vague_goal_signal(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(
        token in lowered
        for token in (
            "que dice",
            "qué dice",
            "explica",
            "hablame",
            "háblame",
            "cuentame",
            "cuéntame",
            "dime",
        )
    )


def _estimate_subqueries(
    *,
    profile: AgentProfile | None,
    intent_mode: str,
    scope_count_requested: int,
) -> int:
    if profile is None:
        return 1
    mode_cfg = profile.query_modes.modes.get(str(intent_mode or "").strip())
    if mode_cfg is None:
        return max(1, scope_count_requested)
    policy = (
        mode_cfg.decomposition_policy if isinstance(mode_cfg.decomposition_policy, dict) else {}
    )
    max_subqueries = int(policy.get("max_subqueries") or 1)
    if scope_count_requested >= 2:
        return max(2, min(max_subqueries, scope_count_requested + 1))
    return max(1, min(max_subqueries, 2 if max_subqueries >= 2 else 1))


def _mode_policy(profile: AgentProfile | None, intent_mode: str) -> ModeInteractionPolicy:
    if profile is None:
        return ModeInteractionPolicy()
    policy = profile.interaction_policy.mode_overrides.get(str(intent_mode or "").strip())
    if isinstance(policy, ModeInteractionPolicy):
        return policy
    return ModeInteractionPolicy()


def decide_interaction(
    *,
    query: str,
    intent: QueryIntent,
    retrieval_plan: RetrievalPlan,
    reasoning_plan: ReasoningPlan,
    profile: AgentProfile | None,
    prior_interruptions: int,
) -> InteractionDecision:
    if profile is None or not bool(profile.interaction_policy.enabled):
        return InteractionDecision(
            level="L1",
            needs_interrupt=False,
            kind="none",
            question="",
            options=(),
            metrics={},
            missing_slots=(),
            scope_candidates=(),
        )

    policy = profile.interaction_policy
    thresholds = policy.thresholds
    mode_policy = _mode_policy(profile, intent.mode)

    requested_scopes = tuple(retrieval_plan.requested_standards or ())
    scope_candidates = detect_scope_candidates(query, profile=profile)
    clarification_round = _clarification_round(query)
    clarification_choice = _clarification_choice(query)
    clarification_confirmed = _clarification_confirmed(query)
    clarification_text = _clarification_text(query)
    scope_count_requested = len(scope_candidates or requested_scopes)
    scope_count_confirmed = len(requested_scopes)
    objective_hint = (
        clarification_text
        if clarification_text and not _looks_like_scope_phrase(clarification_text)
        else ""
    )

    missing_required_slots = 0
    required_slots = [str(v).strip().lower() for v in mode_policy.required_slots if str(v).strip()]
    if "scope" in required_slots and scope_count_confirmed == 0:
        missing_required_slots += 1
    if "objective" in required_slots and len(str(query or "").split()) < 8:
        missing_required_slots += 1

    ambiguity_score = 0.0
    if missing_required_slots > 0:
        ambiguity_score += 0.35
    if scope_count_requested >= 2 and scope_count_confirmed == 0:
        ambiguity_score += 0.25
    if _vague_goal_signal(query):
        ambiguity_score += 0.1
    if re.search(r"\biso\b", str(query or ""), flags=re.IGNORECASE) and scope_count_confirmed == 0:
        ambiguity_score += 0.2
    ambiguity_score = max(0.0, min(1.0, ambiguity_score))

    estimated_subqueries = _estimate_subqueries(
        profile=profile,
        intent_mode=intent.mode,
        scope_count_requested=scope_count_requested,
    )
    estimated_latency_s = round(
        3.2 + (estimated_subqueries * 1.6) + (len(reasoning_plan.steps) * 0.9), 2
    )
    estimated_cost_tokens = int(
        900 + (estimated_subqueries * 1500) + (len(reasoning_plan.steps) * 600)
    )
    coverage_confidence = max(
        0.0,
        min(
            1.0,
            1.0
            - (ambiguity_score * 0.55)
            - (0.25 if (scope_count_requested >= 2 and scope_count_confirmed == 0) else 0.0),
        ),
    )

    risk_level = str(mode_policy.risk_level or "low").lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "low"

    metrics = {
        "ambiguity_score": round(float(ambiguity_score), 4),
        "scope_count_requested": int(scope_count_requested),
        "scope_count_confirmed": int(scope_count_confirmed),
        "missing_required_slots": int(missing_required_slots),
        "estimated_subqueries": int(estimated_subqueries),
        "estimated_cost_tokens": int(estimated_cost_tokens),
        "estimated_latency_s": float(estimated_latency_s),
        "risk_level": risk_level,
        "coverage_confidence": round(float(coverage_confidence), 4),
        "clarification_round": clarification_round,
        "slots_filled": int(scope_count_confirmed),
        "loop_prevented": False,
        "objective_hint_present": bool(objective_hint),
    }

    if prior_interruptions >= int(policy.max_interruptions_per_turn):
        return InteractionDecision(
            level="L1",
            needs_interrupt=False,
            kind="none",
            question="",
            options=(),
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    if _query_has_user_feedback(query):
        return InteractionDecision(
            level="L1",
            needs_interrupt=False,
            kind="none",
            question="",
            options=(),
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    if clarification_confirmed and objective_hint and scope_count_confirmed == 0:
        metrics["loop_prevented"] = True
        metrics["proposal_confirmed_without_scope"] = True
        return InteractionDecision(
            level="L1",
            needs_interrupt=False,
            kind="none",
            question="",
            options=(),
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    if clarification_round >= 2 and scope_count_confirmed == 0:
        metrics["loop_prevented"] = True
        return InteractionDecision(
            level="L1",
            needs_interrupt=False,
            kind="none",
            question="",
            options=(),
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    needs_l2 = (
        missing_required_slots > 0
        or ambiguity_score >= float(thresholds.l2_ambiguity)
        or (scope_count_requested >= 2 and scope_count_confirmed == 0)
        or (coverage_confidence < float(thresholds.low_coverage))
    )

    l3_signals = 0
    l3_signals += 1 if estimated_subqueries >= int(thresholds.l3_subqueries) else 0
    l3_signals += 1 if estimated_latency_s >= float(thresholds.l3_latency_s) else 0
    l3_signals += 1 if estimated_cost_tokens >= int(thresholds.l3_cost_tokens) else 0
    l3_signals += 1 if scope_count_requested >= 3 else 0
    l3_signals += 1 if risk_level == "high" else 0
    if bool(mode_policy.require_plan_approval):
        l3_signals += 1
    needs_l3 = l3_signals >= 2 or (bool(mode_policy.require_plan_approval) and risk_level == "high")

    if needs_l3:
        plan_steps = [
            f"{idx + 1}) {step.tool}" for idx, step in enumerate(reasoning_plan.steps[:4])
        ]
        step_text = " | ".join(plan_steps) if plan_steps else "1) semantic_retrieval"
        question = (
            "Entiendo que quieres un analisis amplio. "
            f"Plan propuesto: {step_text}. "
            "¿Te parece bien este plan o quieres ajustarlo?"
        )
        return InteractionDecision(
            level="L3",
            needs_interrupt=True,
            kind="plan_approval",
            question=question,
            options=("si", "ajustar", "cambiar alcance"),
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    if needs_l2:
        if (
            clarification_choice in {"compare_multiple", "comparar_multiples"}
            and scope_count_confirmed == 0
        ):
            question = (
                "Perfecto, comparemos multiples alcances. "
                "Escribe los alcances exactos separados por coma (ej: ISO 9001, ISO 14001)."
            )
            opts = tuple(scope_candidates[:4]) if scope_candidates else ("Escribir alcances ahora",)
            metrics["guided_reprompt"] = True
        elif scope_count_requested >= 2 and scope_count_confirmed == 0 and scope_candidates:
            opts = tuple(scope_candidates[:4])
            question = f"Veo ambiguedad de alcance. ¿Quieres que responda para: {', '.join(opts)}?"
        elif str(intent.mode or "").strip() in {"cross_scope_analysis", "cross_standard_analysis"}:
            if objective_hint:
                question = (
                    f"Entendi que quieres comparar por '{objective_hint}'. "
                    "Propongo continuar con comparacion multialcance. "
                    "¿Confirmas? Si prefieres acotar, escribe normas exactas separadas por coma."
                )
                opts = ("si, continuar",)
                metrics["proposal_generated"] = True
            else:
                question = (
                    "Para comparar con evidencia util, dime los alcances exactos a incluir "
                    "(ej: ISO 9001, ISO 14001, ISO 45001)."
                )
                opts = tuple(scope_candidates[:4]) if scope_candidates else ()
        else:
            question = (
                "Necesito un dato concreto para responder con evidencia: "
                "indica el alcance exacto que deseas analizar."
            )
            opts = tuple(scope_candidates[:4]) if scope_candidates else ()
        return InteractionDecision(
            level="L2",
            needs_interrupt=True,
            kind="clarification",
            question=question,
            options=opts,
            metrics=metrics,
            missing_slots=tuple(required_slots),
            scope_candidates=tuple(scope_candidates),
        )

    return InteractionDecision(
        level="L1",
        needs_interrupt=False,
        kind="none",
        question="",
        options=(),
        metrics=metrics,
        missing_slots=tuple(required_slots),
        scope_candidates=tuple(scope_candidates),
    )
