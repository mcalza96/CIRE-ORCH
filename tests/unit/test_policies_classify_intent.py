from app.agent.policies import classify_intent


def test_classify_intent_multi_standard_cross_impact_prefers_comparativa() -> None:
    query = (
        "En el contexto de la Gestion del Cambio, como obliga la ISO 45001:2018 (8.1.2) "
        "a reevaluar la ISO 14001:2015 (8.1) y que impacto documental genera en ISO 9001:2015 (8.5.1)?"
    )
    intent = classify_intent(query)
    assert intent.mode == "comparativa"

