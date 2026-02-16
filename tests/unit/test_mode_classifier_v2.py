from app.agent.mode_classifier import classify_mode_v2


def test_multiclause_analytical_prompt_is_not_literal() -> None:
    query = (
        "Analice cómo la falta de definición de la responsabilidad y autoridad (Cláusula 5.3) "
        "impacta la consulta y participación (Cláusula 5.4) e impide demostrar la mejora continua (Cláusula 10.3). "
        "ISO 9001 9.1.2 e ISO 14001 9.1.1."
    )
    result = classify_mode_v2(query)
    assert result.mode in {"comparativa", "explicativa"}
    assert "literal_normativa" in set(result.blocked_modes)


def test_clause_without_scope_prefers_not_ambigua_here() -> None:
    query = "Transcribe el texto exacto de la cláusula 8.4.3"
    result = classify_mode_v2(query)
    # v2 mode classifier alone doesn't do scope clarification; policies layer handles ambigua_scope.
    assert result.mode in {"literal_normativa", "literal_lista", "explicativa", "comparativa"}
