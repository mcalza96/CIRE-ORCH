from app.core.cli_utils import rewrite_query_with_clarification


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
