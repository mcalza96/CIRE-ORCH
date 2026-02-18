from app.agent.policies import (
    build_retrieval_plan,
    classify_intent,
    detect_conflict_objectives,
    detect_scope_candidates,
    suggest_scope_candidates,
)
from app.cartridges.builtin_profiles import BUILTIN_PROFILES
from app.cartridges.models import AgentProfile

# Cargamos el perfil iso_auditor para los tests que dependen de patrones ISO
ISO_PROFILE = AgentProfile.model_validate(BUILTIN_PROFILES["iso_auditor"])


def test_classify_intent_literal_lista():
    intent = classify_intent(
        "Lista las entradas exclusivas de la cláusula 9.1", profile=ISO_PROFILE
    )
    assert intent.mode == "literal_list_extract"


def test_classify_intent_comparativa():
    intent = classify_intent(
        "Compara ISO 27001 vs ISO 9001 para proveedores externos", profile=ISO_PROFILE
    )
    assert intent.mode == "cross_standard_analysis"


def test_build_retrieval_plan_literal_is_strict():
    query = "Que documento obligatorio exige ISO 9001 en la clausula 6.1.3"
    intent = classify_intent(query, profile=ISO_PROFILE)
    plan = build_retrieval_plan(intent, query=query, profile=ISO_PROFILE)
    assert plan.require_literal_evidence is True
    # Validamos que los parametros sean robustos para busqueda literal
    assert plan.chunk_k >= 30
    assert plan.summary_k <= 5


def test_classify_intent_ambiguous_scope_without_standard():
    intent = classify_intent(
        "__mode__=ambigua_scope Que exige la clausula 9.1.2?", profile=ISO_PROFILE
    )
    assert intent.mode == "scope_ambiguity"


def test_suggest_scope_candidates_uses_domain_hints():
    options = suggest_scope_candidates(
        "requisitos legales ambientales de la clausula 9.1.2", profile=ISO_PROFILE
    )
    assert "ISO 14001" in options


def test_detect_scope_candidates_supports_bare_iso_numbers():
    options = detect_scope_candidates(
        "impacta 9001, 14001 y 45001 por ciberataque", profile=ISO_PROFILE
    )
    # Orden lexicografico: 14001, 45001, 9001
    assert set(options) == {"ISO 9001", "ISO 14001", "ISO 45001"}


def test_detect_conflict_objectives_for_whistleblower_case():
    assert (
        detect_conflict_objectives(
            "Existe conflicto entre trazabilidad de evidencia y confidencialidad de denuncia anonima",
            profile=ISO_PROFILE,
        )
        is True
    )
