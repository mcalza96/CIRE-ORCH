from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agent.types.models import QueryIntent, ReasoningPlan, RetrievalPlan
from app.agent.policies import detect_scope_candidates
from app.profiles.models import AgentProfile, ModeInteractionPolicy


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


def _query_has_user_feedback(query: str, context: dict[str, Any] | None = None) -> bool:
    if isinstance(context, dict):
        if bool(context.get("plan_approved")):
            return True
        if str(context.get("plan_feedback") or "").strip():
            return True
    lowered = str(query or "").lower()
    markers = (
        "__plan_approved__=true",
        "__plan_feedback__=",
    )
    return any(marker in lowered for marker in markers)


def _clarification_round(query: str, context: dict[str, Any] | None = None) -> int:
    if isinstance(context, dict):
        raw = context.get("round")
        try:
            return max(0, int(raw or 0))
        except Exception:
            pass
    match = re.search(r"__clarification_round__=(\d+)", str(query or ""), flags=re.IGNORECASE)
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def _clarification_choice(query: str, context: dict[str, Any] | None = None) -> str:
    if isinstance(context, dict):
        selected = str(context.get("selected_option") or "").strip().lower()
        if selected:
            return selected
    match = re.search(
        r"__clarification_choice__=([a-z0-9_\-]{2,64})",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return str(match.group(1) if match else "").strip().lower()


def _clarification_confirmed(query: str, context: dict[str, Any] | None = None) -> bool:
    if isinstance(context, dict):
        if bool(context.get("confirmed")):
            return True
        if bool(context.get("plan_approved")):
            return True
    return bool(
        re.search(r"__clarification_confirmed__=true", str(query or ""), flags=re.IGNORECASE)
    )


def _clarification_text(query: str, context: dict[str, Any] | None = None) -> str:
    if isinstance(context, dict):
        text = str(context.get("answer_text") or "").strip()
        if text:
            return text
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
    compact = re.sub(r"\s+", " ", value)
    if re.search(
        r"^(?:iso|iec|nom|nmx|nfpa|osha|en|une|iram|bs|din)\s*[-:_]?\s*\d{2,6}(?:[:\-]\d{4})?$",
        compact,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"^[A-Za-z]{2,12}[-_ ]?\d{2,6}$", compact):
        return True
    return bool(re.search(r"^\d{3,6}$", compact))


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


def _requested_scopes_from_context(context: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(context, dict):
        return ()
    raw = context.get("requested_scopes")
    if not isinstance(raw, list):
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw:
        value = str(item or "").strip()
        if not value:
            continue
        normalized = value.upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


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
    clarification_context: dict[str, Any] | None = None,
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

    context_scopes = _requested_scopes_from_context(clarification_context)
    requested_scopes = tuple(retrieval_plan.requested_standards or context_scopes)
    detected_candidates = detect_scope_candidates(query, profile=profile)
    merged_candidates: list[str] = []
    seen_candidates: set[str] = set()
    for scope in tuple(context_scopes) + tuple(detected_candidates):
        value = str(scope or "").strip()
        if not value:
            continue
        key = value.upper()
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        merged_candidates.append(key)
    scope_candidates = tuple(merged_candidates)
    clarification_round = _clarification_round(query, clarification_context)
    clarification_choice = _clarification_choice(query, clarification_context)
    clarification_confirmed = _clarification_confirmed(query, clarification_context)
    clarification_text = _clarification_text(query, clarification_context)
    scope_count_requested = len(scope_candidates or requested_scopes)
    scope_count_confirmed = len(requested_scopes)
    objective_hint_context = (
        str((clarification_context or {}).get("objective_hint") or "").strip()
        if isinstance(clarification_context, dict)
        else ""
    )
    objective_hint = objective_hint_context or (
        clarification_text
        if clarification_text and not _looks_like_scope_phrase(clarification_text)
        else ""
    )

    required_slots = [str(v).strip().lower() for v in mode_policy.required_slots if str(v).strip()]
    missing_slots_list: list[str] = []

    if "scope" in required_slots and scope_count_confirmed == 0:
        missing_slots_list.append("scope")

    objective_extracted = False
    if isinstance(clarification_context, dict):
        obj_list = clarification_context.get("objective")
        if isinstance(obj_list, list) and obj_list:
            objective_extracted = True

    if "objective" in required_slots and not objective_extracted and len(str(query or "").split()) < 8:
        missing_slots_list.append("objective")

    missing_required_slots = len(missing_slots_list)

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

    needs_l2 = bool(
        missing_required_slots > 0
        or (
            ambiguity_score >= float(thresholds.l2_ambiguity)
            and scope_count_requested >= 1
            and scope_count_confirmed == 0
        )
    )
    needs_l3 = bool(
        mode_policy.require_plan_approval
        or estimated_subqueries >= int(thresholds.l3_subqueries)
        or estimated_latency_s >= float(thresholds.l3_latency_s)
        or estimated_cost_tokens >= int(thresholds.l3_cost_tokens)
        or (risk_level == "high" and ambiguity_score >= float(thresholds.l2_ambiguity))
    )

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

    level = "L1"
    needs_interrupt = False
    kind = "none"
    question = ""
    options: tuple[str, ...] = ()

    if prior_interruptions >= int(policy.max_interruptions_per_turn):
        pass
    elif _query_has_user_feedback(query, clarification_context):
        pass
    elif clarification_confirmed and objective_hint and scope_count_confirmed == 0:
        metrics["loop_prevented"] = True
        metrics["proposal_confirmed_without_scope"] = True
    elif clarification_round >= 2 and scope_count_confirmed == 0:
        metrics["loop_prevented"] = True
    elif needs_l3:
        _TOOL_DISPLAY_NAMES = {
            "semantic_retrieval": "Buscar contexto normativo",
            "logical_comparison": "Analizar cruces y vacios logicos",
            "structural_extraction": "Extraer y estructurar datos",
            "python_calculator": "Ejecutar calculos matematicos",
            "citation_validator": "Validar citas contra la fuente",
        }
        consolidated_steps = []
        for step in reasoning_plan.steps:
            name = _TOOL_DISPLAY_NAMES.get(step.tool, step.tool)
            if consolidated_steps and consolidated_steps[-1]["name"] == name:
                consolidated_steps[-1]["count"] += 1
            else:
                consolidated_steps.append({"name": name, "count": 1})
        
        plan_steps = []
        for idx, item in enumerate(consolidated_steps[:4]):
            if item["count"] > 1:
                plan_steps.append(f"{idx + 1}) {item['name']} ({item['count']}x paralelizado)")
            else:
                plan_steps.append(f"{idx + 1}) {item['name']}")

        step_text = " -> ".join(plan_steps) if plan_steps else "1) Buscar contexto normativo"
        question = (
            "Entiendo que requieres un analisis profundo. "
            f"Plan propuesto: {step_text}. "
            "¿Te parece bien este plan o quieres ajustarlo (ej: pedir enfoque en una tabla)?"
        )
        level = "L3"
        needs_interrupt = True
        kind = "plan_approval"
        options = ("si", "ajustar", "cambiar alcance")
    elif needs_l2:
        level = "L2"
        needs_interrupt = True
        kind = "clarification"
        if (
            clarification_choice in {"compare_multiple", "comparar_multiples"}
            and scope_count_confirmed == 0
        ):
            dynamic_example = (
                ", ".join(scope_candidates[:2]) if scope_candidates else "alcance A, alcance B"
            )
            question = (
                "Perfecto, comparemos multiples alcances. "
                f"Escribe los alcances exactos separados por coma (ej: {dynamic_example})."
            )
            options = (
                tuple(scope_candidates[:4]) if scope_candidates else ("Escribir alcances ahora",)
            )
            metrics["guided_reprompt"] = True
        elif scope_count_requested >= 2 and scope_count_confirmed == 0 and scope_candidates:
            options = tuple(scope_candidates[:4])
            question = (
                f"Veo ambiguedad de alcance. ¿Quieres que responda para: {', '.join(options)}?"
            )
        elif str(intent.mode or "").strip() in {"cross_scope_analysis", "cross_standard_analysis"}:
            if objective_hint:
                question = (
                    f"Entendi que quieres comparar por '{objective_hint}'. "
                    "Propongo continuar con comparacion multialcance. "
                    "¿Confirmas? Si prefieres acotar, escribe normas exactas separadas por coma."
                )
                options = ("si, continuar",)
                metrics["proposal_generated"] = True
            else:
                dynamic_example = (
                    ", ".join(scope_candidates[:3]) if scope_candidates else "alcance A, alcance B"
                )
                question = (
                    "Para comparar con evidencia util, dime los alcances exactos a incluir "
                    f"(ej: {dynamic_example})."
                )
                options = tuple(scope_candidates[:4]) if scope_candidates else ()
        else:
            question = (
                "Necesito un dato concreto para responder con evidencia: "
                "indica el alcance exacto que deseas analizar."
            )
            options = tuple(scope_candidates[:4]) if scope_candidates else ()

    return InteractionDecision(
        level=level,
        needs_interrupt=needs_interrupt,
        kind=kind,
        question=question,
        options=options,
        metrics=metrics,
        missing_slots=tuple(missing_slots_list),
        scope_candidates=tuple(scope_candidates),
    )
