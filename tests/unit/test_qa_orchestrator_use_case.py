from dataclasses import dataclass
import asyncio
import pytest

from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan, ValidationResult, QueryIntent, ClarificationRequest
from app.cartridges.models import AgentProfile, RouterHeuristics, ScopePattern

# Mocks deterministas para build_retrieval_plan y funciones de clasificación
def _mock_classify_trace(q, profile=None):
    if "clausula 9.1.2" in q.lower():
        return QueryIntent(mode="ambigua_scope"), {"confidence": 1.0}
    if "impacto" in q.lower():
        return QueryIntent(mode="explicativa"), {"confidence": 1.0}
    return QueryIntent(mode="literal_normativa"), {"confidence": 1.0}

def _mock_plan(intent, query="", profile=None):
    return RetrievalPlan(
        mode=intent.mode,
        chunk_k=10,
        chunk_fetch_k=40,
        summary_k=2,
        require_literal_evidence=(intent.mode == "literal_normativa")
    )

@dataclass
class _FakeRetriever:
    async def retrieve_chunks(self, *args, **kwargs):
        return [EvidenceItem(source="C1", content="[6.1.3 d)] Declaracion de aplicabilidad...")]

    async def retrieve_summaries(self, *args, **kwargs):
        return [EvidenceItem(source="R1", content="Resumen regulatorio")]

@dataclass
class _FakeAnswerGenerator:
    async def generate(self, *args, **kwargs):
        return AnswerDraft(text="6.1.3 d) | 'Declaracion de aplicabilidad' | Fuente(C1)", mode="literal_normativa", evidence=[])

@dataclass
class _FakeValidator:
    def validate(self, *args, **kwargs):
        return ValidationResult(accepted=True, issues=[])

@pytest.mark.asyncio
async def test_use_case_executes_and_returns_validated_answer(monkeypatch):
    import app.agent.application as app_mod
    monkeypatch.setattr(app_mod, "classify_intent_with_trace", _mock_classify_trace)
    monkeypatch.setattr(app_mod, "build_retrieval_plan", _mock_plan)

    from app.agent.application import HandleQuestionUseCase, HandleQuestionCommand
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    result = await use_case.execute(HandleQuestionCommand(
        query="Que exige ISO 9001 en 6.1.3?", tenant_id="t1", collection_id="c1", scope_label="s1"
    ))
    assert result.intent.mode == "literal_normativa"
    assert "Fuente(C1)" in result.answer.text

@pytest.mark.asyncio
async def test_use_case_requests_scope_disambiguation_for_ambiguous_clause(monkeypatch):
    import app.agent.application as app_mod
    monkeypatch.setattr(app_mod, "classify_intent_with_trace", _mock_classify_trace)
    monkeypatch.setattr(app_mod, "build_retrieval_plan", _mock_plan)

    # Inyectamos regla de desambiguación para modo ambigua_scope
    test_profile = AgentProfile(
        profile_id="test",
        router=RouterHeuristics(reference_patterns=[r"\b\d+(?:\.\d+)+\b"]),
        clarification_rules=[{
            "rule_id": "ambiguity_rule",
            "all_markers": ["__mode__=ambigua_scope"],
            "question_template": "necesito desambiguar alcances: {scopes}",
            "options": []
        }]
    )

    from app.agent.application import HandleQuestionUseCase, HandleQuestionCommand
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    result = await use_case.execute(HandleQuestionCommand(
        query="Que exige la clausula 9.1.2?", tenant_id="t1", collection_id="c1", scope_label="s1", agent_profile=test_profile
    ))
    assert result.intent.mode == "ambigua_scope"
    assert "necesito desambiguar" in result.answer.text.lower()

@pytest.mark.asyncio
async def test_use_case_requests_clarification_for_multi_scope_explanatory_query(monkeypatch):
    import app.agent.application as app_mod
    monkeypatch.setattr(app_mod, "classify_intent_with_trace", _mock_classify_trace)
    monkeypatch.setattr(app_mod, "build_retrieval_plan", _mock_plan)
    monkeypatch.setattr(app_mod, "detect_scope_candidates", lambda q, profile=None: ["ISO 9001", "ISO 14001"])

    test_profile = AgentProfile(
        profile_id="test",
        clarification_rules=[{
            "rule_id": "multi_scope",
            "mode": "explicativa",
            "min_scope_count": 2,
            "question_template": "multiples alcances detected: {scopes}",
            "options": ["Analisis integrado"]
        }]
    )

    from app.agent.application import HandleQuestionUseCase, HandleQuestionCommand
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    result = await use_case.execute(HandleQuestionCommand(
        query="Impacto en ISO 9001 y 14001", tenant_id="t1", collection_id="c1", scope_label="s1", agent_profile=test_profile
    ))
    assert result.clarification is not None
    assert "multiples alcances" in result.clarification.question.lower()
