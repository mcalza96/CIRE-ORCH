from app.agent.policies import (
    build_retrieval_plan,
    classify_intent,
    detect_conflict_objectives,
    detect_scope_candidates,
    suggest_scope_candidates,
)


def test_classify_intent_literal_lista():
    intent = classify_intent("Lista las entradas exclusivas de la cláusula 9.3")
    assert intent.mode == "literal_lista"


def test_classify_intent_comparativa():
    intent = classify_intent("Compara ISO 27001 vs ISO 9001 para proveedores externos")
    assert intent.mode == "comparativa"


def test_build_retrieval_plan_literal_is_strict():
    query = "Que documento obligatorio exige ISO 9001 en la clausula 6.1.3"
    intent = classify_intent(query)
    plan = build_retrieval_plan(intent, query=query)
    assert plan.require_literal_evidence is True
    assert plan.chunk_k >= 40
    assert plan.summary_k <= 3


def test_classify_intent_ambiguous_scope_without_standard():
    intent = classify_intent("Que exige la clausula 9.1.2?")
    assert intent.mode == "ambigua_scope"


def test_suggest_scope_candidates_uses_domain_hints():
    options = suggest_scope_candidates("requisitos legales ambientales de la clausula 9.1.2")
    assert options[0] == "ISO 14001"


def test_detect_scope_candidates_supports_bare_iso_numbers():
    options = detect_scope_candidates("impacta 9001, 14001 y 45001 por ciberataque")
    assert options == ("ISO 9001", "ISO 14001", "ISO 45001")


def test_detect_conflict_objectives_for_whistleblower_case():
    assert detect_conflict_objectives(
        "Existe conflicto entre trazabilidad de evidencia y confidencialidad de denuncia anonima"
    ) is True
