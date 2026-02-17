from app.agent.policies import classify_intent_with_trace
from app.cartridges.models import AgentProfile, RouterHeuristics


def test_classify_intent_with_trace_multiscope_multiclause_is_comparativa_or_explicativa() -> None:
    query = (
        "Analice cómo la falta de responsabilidad (Cláusula 5.3) impacta la participación (ISO 45001 5.4) "
        "y cómo impide la mejora continua (10.3) basándose en datos de satisfacción (ISO 9001 9.1.2) "
        "y desempeño ambiental (ISO 14001 9.1.1)."
    )
    intent, trace = classify_intent_with_trace(query)
    assert intent.mode in {"comparativa", "explicativa"}
    assert isinstance(trace, dict)
    assert trace.get("version") == "v2"
    assert float(trace.get("confidence") or 0.0) > 0.4


def test_classify_intent_with_trace_low_signal_defaults_to_explicativa() -> None:
    profile = AgentProfile(
        profile_id="iso_like",
        router=RouterHeuristics(
            literal_normative_hints=["textualmente", "verbatim"],
            literal_list_hints=["lista", "enumera"],
            comparative_hints=["comparar", "vs"],
            interpretive_hints=["analiza", "impacta"],
        ),
    )
    intent, trace = classify_intent_with_trace(
        "que dice la introduccion de la iso 9001?",
        profile=profile,
    )
    assert intent.mode == "explicativa"
    assert trace.get("version") == "v2"
