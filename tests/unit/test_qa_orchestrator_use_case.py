from dataclasses import dataclass
import asyncio

from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan, ValidationResult


@dataclass
class _FakeRetriever:
    async def retrieve_chunks(self, query: str, tenant_id: str, collection_id: str | None, plan: RetrievalPlan):
        return [EvidenceItem(source="C1", content="[6.1.3 d)] Declaracion de aplicabilidad...")]

    async def retrieve_summaries(self, query: str, tenant_id: str, collection_id: str | None, plan: RetrievalPlan):
        return [EvidenceItem(source="R1", content="Resumen regulatorio")]


@dataclass
class _FakeAnswerGenerator:
    async def generate(self, query: str, scope_label: str, plan: RetrievalPlan, chunks, summaries):
        text = "6.1.3 d) | 'Declaracion de aplicabilidad' | Fuente(C1)"
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])


@dataclass
class _FakeValidator:
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str):
        return ValidationResult(accepted=True, issues=[])


@dataclass
class _FakeMismatchValidator:
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str):
        return ValidationResult(accepted=False, issues=["Scope mismatch detected: evidence includes sources outside requested standard scope."])


def test_use_case_executes_and_returns_validated_answer():
    use_case = HandleQuestionUseCase(
        retriever=_FakeRetriever(),
        answer_generator=_FakeAnswerGenerator(),
        validator=_FakeValidator(),
    )

    result = asyncio.run(
        use_case.execute(
            HandleQuestionCommand(
                query="Que documento obligatorio exige ISO 9001 en la clausula 6.1.3?",
                tenant_id="tenant-1",
                collection_id="collection-uuid",
                scope_label="tenant=tenant-1 / coleccion=iso",
            )
        )
    )

    assert result.intent.mode == "literal_normativa"
    assert result.plan.require_literal_evidence is True
    assert result.validation.accepted is True
    assert "Fuente(C1)" in result.answer.text


def test_use_case_blocks_answer_on_scope_mismatch():
    use_case = HandleQuestionUseCase(
        retriever=_FakeRetriever(),
        answer_generator=_FakeAnswerGenerator(),
        validator=_FakeMismatchValidator(),
    )

    result = asyncio.run(
        use_case.execute(
            HandleQuestionCommand(
                query="Que documento obligatorio exige ISO 9001 en la clausula 6.1.3?",
                tenant_id="tenant-1",
                collection_id="collection-uuid",
                scope_label="tenant=tenant-1 / coleccion=iso",
            )
        )
    )

    assert result.validation.accepted is False
    assert "respuesta bloqueada" in result.answer.text.lower()


def test_use_case_requests_scope_disambiguation_for_ambiguous_clause():
    use_case = HandleQuestionUseCase(
        retriever=_FakeRetriever(),
        answer_generator=_FakeAnswerGenerator(),
        validator=_FakeValidator(),
    )

    result = asyncio.run(
        use_case.execute(
            HandleQuestionCommand(
                query="Que exige la clausula 9.1.2?",
                tenant_id="tenant-1",
                collection_id="collection-uuid",
                scope_label="tenant=tenant-1 / coleccion=iso",
            )
        )
    )

    assert result.intent.mode == "ambigua_scope"
    assert "indica la norma objetivo" in result.answer.text.lower()


def test_use_case_requests_clarification_for_multi_scope_explanatory_query():
    use_case = HandleQuestionUseCase(
        retriever=_FakeRetriever(),
        answer_generator=_FakeAnswerGenerator(),
        validator=_FakeValidator(),
    )

    result = asyncio.run(
        use_case.execute(
            HandleQuestionCommand(
                query=(
                    "Durante una auditoría remota, un incidente afecta emisiones y alertas de seguridad. "
                    "Evalúa el impacto documental en ISO 9001."
                ),
                tenant_id="tenant-1",
                collection_id="collection-uuid",
                scope_label="tenant=tenant-1 / coleccion=iso",
            )
        )
    )

    assert result.clarification is not None
    assert "múltiples normas" in result.clarification.question.lower()


def test_use_case_requests_clarification_for_conflict_mode():
    use_case = HandleQuestionUseCase(
        retriever=_FakeRetriever(),
        answer_generator=_FakeAnswerGenerator(),
        validator=_FakeValidator(),
    )

    result = asyncio.run(
        use_case.execute(
            HandleQuestionCommand(
                query=(
                    "Hay conflicto entre confidencialidad de denuncia anónima y trazabilidad de evidencia "
                    "en ISO 9001, 14001 y 45001."
                ),
                tenant_id="tenant-1",
                collection_id="collection-uuid",
                scope_label="tenant=tenant-1 / coleccion=iso",
            )
        )
    )

    assert result.clarification is not None
    assert "conflicto" in result.clarification.question.lower()
