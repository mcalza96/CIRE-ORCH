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

    result = validator.validate(draft=draft, plan=plan, query=query)

    assert result.accepted is True
    assert not any("Literal clause coverage insufficient" in issue for issue in result.issues)
