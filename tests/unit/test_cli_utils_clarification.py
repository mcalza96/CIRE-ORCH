from app.core.cli_utils import (
    build_clarification_context,
    extract_scope_list_from_answer,
    rewrite_query_with_clarification,
)


def test_rewrite_clarification_adds_round_and_choice_tags() -> None:
    query = "compara normas"
    rewritten = rewrite_query_with_clarification(
        query,
        "comparar multiples",
        clarification_kind="clarification",
    )
    assert "__clarification_round__=1" in rewritten
    assert "__clarification_choice__=compare_multiple" in rewritten
    assert "__plan_feedback__" not in rewritten


def test_rewrite_plan_approval_uses_plan_feedback_tags() -> None:
    query = "analiza contrato"
    rewritten = rewrite_query_with_clarification(
        query,
        "si",
        clarification_kind="plan_approval",
    )
    assert "__plan_approved__=true" in rewritten


def test_rewrite_clarification_confirmation_tag() -> None:
    query = "compara"
    rewritten = rewrite_query_with_clarification(
        query,
        "si, continuar",
        clarification_kind="clarification",
    )
    assert "__clarification_confirmed__=true" in rewritten


def test_rewrite_scope_list_sets_requested_scopes_marker() -> None:
    query = "compara"
    rewritten = rewrite_query_with_clarification(
        query,
        "ISO 9001, ISO 14001, ISO 45001",
        clarification_kind="clarification",
    )
    assert "__requested_scopes__=[ISO 9001|ISO 14001|ISO 45001]" in rewritten


def test_build_clarification_context_extracts_scope_slots() -> None:
    context = build_clarification_context(
        clarification={"kind": "clarification", "missing_slots": ["scope"]},
        answer_text="ISO 9001, ISO 14001",
        round_no=2,
    )
    assert context["round"] == 2
    assert context["requested_scopes"] == ["ISO 9001", "ISO 14001"]
    assert "objective_hint" not in context


def test_build_clarification_context_uses_objective_hint_when_no_scope() -> None:
    context = build_clarification_context(
        clarification={"kind": "clarification", "missing_slots": ["scope"]},
        answer_text="objetivo: comparar implicaciones operativas",
        round_no=1,
    )
    assert context["requested_scopes"] == []
    assert context.get("objective_hint") == "objetivo: comparar implicaciones operativas"


def test_extract_scope_list_rejects_generic_phrase_with_number() -> None:
    scopes = extract_scope_list_from_answer("los objetivos de las 3 normas")
    assert scopes == []
