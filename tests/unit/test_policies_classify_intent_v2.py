from app.agent.policies import classify_intent_with_trace


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
