from app.agent.adapters import LiteralEvidenceValidator
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan
from app.core.config import settings


def _build_plan() -> RetrievalPlan:
    return RetrievalPlan(
        mode="literal_normativa",
        chunk_k=8,
        chunk_fetch_k=20,
        summary_k=3,
        require_literal_evidence=True,
        requested_standards=("ISO 9001",),
    )


def test_literal_clause_mismatch_uses_semantic_fallback() -> None:
    validator = LiteralEvidenceValidator()
    plan = _build_plan()
    query = "Que exige ISO 9001 en la clausula 7.5.3 sobre integridad de informacion documentada?"
    row = {
        "content": "La informacion documentada debe protegerse contra perdida de integridad y uso indebido.",
        "similarity": 0.92,
        "metadata": {
            "source_standard": "ISO 9001",
            "clause_title": "Control de la informacion documentada",
        },
    }
    draft = AnswerDraft(
        text="Hallazgo con evidencia. Fuente(C1)",
        mode=plan.mode,
        evidence=[EvidenceItem(source="C1", content=row["content"], metadata={"row": row})],
    )

    result = validator.validate(draft=draft, plan=plan, query=query)

    assert result.accepted is True
    assert not any("Literal clause mismatch" in issue for issue in result.issues)


def test_literal_clause_mismatch_remains_when_semantic_signal_is_weak() -> None:
    validator = LiteralEvidenceValidator()
    plan = _build_plan()
    query = "Que exige ISO 9001 en la clausula 7.5.3 sobre integridad de informacion documentada?"
    row = {
        "content": "El proveedor debe gestionar las entregas segun requisitos comerciales.",
        "similarity": 0.18,
        "metadata": {
            "source_standard": "ISO 9001",
            "clause_title": "Compras",
        },
    }
    draft = AnswerDraft(
        text="Hallazgo con evidencia. Fuente(C1)",
        mode=plan.mode,
        evidence=[EvidenceItem(source="C1", content=row["content"], metadata={"row": row})],
    )

    result = validator.validate(draft=draft, plan=plan, query=query)

    assert result.accepted is False
    assert any("Literal clause mismatch" in issue for issue in result.issues)


def test_literal_clause_mismatch_when_semantic_fallback_disabled() -> None:
    validator = LiteralEvidenceValidator()
    plan = _build_plan()
    query = "Que exige ISO 9001 en la clausula 7.5.3 sobre integridad de informacion documentada?"
    row = {
        "content": "La informacion documentada debe protegerse contra perdida de integridad y uso indebido.",
        "similarity": 0.99,
        "metadata": {
            "source_standard": "ISO 9001",
            "clause_title": "Control de la informacion documentada",
        },
    }
    draft = AnswerDraft(
        text="Hallazgo con evidencia. Fuente(C1)",
        mode=plan.mode,
        evidence=[EvidenceItem(source="C1", content=row["content"], metadata={"row": row})],
    )

    original = settings.QA_LITERAL_SEMANTIC_FALLBACK_ENABLED
    settings.QA_LITERAL_SEMANTIC_FALLBACK_ENABLED = False
    try:
        result = validator.validate(draft=draft, plan=plan, query=query)
    finally:
        settings.QA_LITERAL_SEMANTIC_FALLBACK_ENABLED = original

    assert result.accepted is False
    assert any("Literal clause mismatch" in issue for issue in result.issues)


def test_literal_clause_coverage_requires_all_for_two_refs() -> None:
    validator = LiteralEvidenceValidator()
    plan = _build_plan()
    query = "Cita 6.1 y 9.1 de ISO 9001"
    row = {
        "content": "[CLAUSE_ID: 6.1] texto de la clausula 6.1",
        "metadata": {"source_standard": "ISO 9001", "clause_id": "6.1"},
    }
    draft = AnswerDraft(
        text="Hallazgo con evidencia. Fuente(C1)",
        mode=plan.mode,
        evidence=[EvidenceItem(source="C1", content=row["content"], metadata={"row": row})],
    )

    result = validator.validate(draft=draft, plan=plan, query=query)

    assert result.accepted is False
    assert any("Literal clause coverage insufficient" in issue for issue in result.issues)


def test_literal_clause_coverage_allows_partial_for_three_plus_refs() -> None:
    validator = LiteralEvidenceValidator()
    plan = _build_plan()
    query = "Cita 6.1, 9.1 y 9.2 de ISO 9001"
    rows = [
        {
            "content": "[CLAUSE_ID: 6.1] texto",
            "metadata": {"source_standard": "ISO 9001", "clause_id": "6.1"},
        },
        {
            "content": "[CLAUSE_ID: 9.1] texto",
            "metadata": {"source_standard": "ISO 9001", "clause_id": "9.1"},
        },
    ]
    draft = AnswerDraft(
        text="Hallazgo con evidencia. Fuente(C1) Fuente(C2)",
        mode=plan.mode,
        evidence=[
            EvidenceItem(source="C1", content=rows[0]["content"], metadata={"row": rows[0]}),
            EvidenceItem(source="C2", content=rows[1]["content"], metadata={"row": rows[1]}),
        ],
    )

    original_ratio = settings.ORCH_LITERAL_REF_MIN_COVERAGE_RATIO
    settings.ORCH_LITERAL_REF_MIN_COVERAGE_RATIO = 0.66
    try:
        result = validator.validate(draft=draft, plan=plan, query=query)
    finally:
        settings.ORCH_LITERAL_REF_MIN_COVERAGE_RATIO = original_ratio

    assert result.accepted is True
    assert not any("Literal clause coverage insufficient" in issue for issue in result.issues)


def test_structured_inference_requires_citations_in_inference_section() -> None:
    validator = LiteralEvidenceValidator()
    plan = RetrievalPlan(
        mode="grounded_inference",
        chunk_k=8,
        chunk_fetch_k=20,
        summary_k=3,
        require_literal_evidence=False,
        requested_standards=("ISO 9001",),
    )
    draft = AnswerDraft(
        text=(
            "## Hechos citados\n- Registro evaluado [C1]\n"
            "## Inferencias\n- Existe riesgo de incumplimiento operativo.\n"
            "## Brechas\n- Falta evidencia de programa.\n"
            "## Recomendaciones\n- Definir plan.\n"
            "## Confianza y supuestos\n- Media."
        ),
        mode=plan.mode,
        evidence=[EvidenceItem(source="C1", content="texto", metadata={"row": {"metadata": {}}})],
    )

    result = validator.validate(draft=draft, plan=plan, query="analiza brechas")

    assert result.accepted is False
    assert any("Grounded inference requires" in issue for issue in result.issues)


def test_strong_claim_without_citation_is_rejected() -> None:
    validator = LiteralEvidenceValidator()
    plan = RetrievalPlan(
        mode="grounded_inference",
        chunk_k=8,
        chunk_fetch_k=20,
        summary_k=3,
        require_literal_evidence=False,
    )
    draft = AnswerDraft(
        text=(
            "## Hechos citados\n- Sin fuente\n"
            "## Inferencias\n- Riesgo critico de incumplimiento.\n"
            "## Brechas\n- No hay evidencia.\n"
            "## Recomendaciones\n- Corregir.\n"
            "## Confianza y supuestos\n- Baja."
        ),
        mode=plan.mode,
        evidence=[],
    )

    result = validator.validate(draft=draft, plan=plan, query="riesgo")

    assert result.accepted is False
    assert any(
        "Strong risk claim without explicit evidence markers" in issue for issue in result.issues
    )


def test_cross_scope_validation_requires_evidence_for_each_requested_scope() -> None:
    validator = LiteralEvidenceValidator()
    plan = RetrievalPlan(
        mode="cross_scope_analysis",
        chunk_k=20,
        chunk_fetch_k=80,
        summary_k=0,
        require_literal_evidence=False,
        requested_standards=("ISO 9001", "ISO 14001", "ISO 45001"),
    )
    draft = AnswerDraft(
        text="Comparativa preliminar con referencia C1.",
        mode=plan.mode,
        evidence=[
            EvidenceItem(
                source="C1",
                content="evidencia de ISO 9001",
                metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
            )
        ],
    )

    result = validator.validate(
        draft=draft, plan=plan, query="compara ISO 9001, ISO 14001 y ISO 45001"
    )

    assert result.accepted is False
    assert any(
        "missing evidence coverage for requested standards" in issue for issue in result.issues
    )
