import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agent.application import HandleQuestionUseCase, HandleQuestionCommand
from app.agent.models import (
    RetrievalPlan,
    AnswerDraft,
    ValidationResult,
    RetrievalDiagnostics,
    EvidenceItem,
    QueryIntent
)

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Behave like a real retriever that stores diagnostics
    retriever.last_retrieval_diagnostics = RetrievalDiagnostics(contract="advanced", strategy="mock")
    # Mock optional async methods
    retriever.validate_scope = AsyncMock(return_value={"valid": True})
    return retriever

@pytest.fixture
def mock_generator():
    generator = MagicMock()
    return generator

@pytest.fixture
def mock_validator():
    validator = MagicMock()
    return validator

@pytest.fixture
def use_case(mock_retriever, mock_generator, mock_validator):
    return HandleQuestionUseCase(mock_retriever, mock_generator, mock_validator)

@pytest.mark.asyncio
async def test_silent_correction_scope_mismatch_regenerates(use_case, mock_retriever, mock_generator, mock_validator):
    # Setup
    cmd = HandleQuestionCommand(
        query="question about ISO 9001",
        tenant_id="t1",
        collection_id="c1",
        scope_label="s1"
    )
    
    # 1. Retrieval returns detailed evidence
    chunks = [
        EvidenceItem(source="C1", content="ISO 9001 info", metadata={"row": {"source_standard": "ISO 9001"}}),
        EvidenceItem(source="C2", content="ISO 14001 info", metadata={"row": {"source_standard": "ISO 14001"}}) # Out of scope
    ]
    mock_retriever.retrieve_chunks = AsyncMock(return_value=chunks)
    mock_retriever.retrieve_summaries = AsyncMock(return_value=[])
    
    # 2. Generator logic: first attempt fails scope check, second succeeds
    mock_generator.generate = AsyncMock(side_effect=[
        AnswerDraft(text="Draft 1 uses ISO 14001", mode="explicativa", evidence=chunks),
        AnswerDraft(text="Draft 2 uses only ISO 9001", mode="explicativa", evidence=[chunks[0]])
    ])
    
    # 3. Validator logic
    mock_validator.validate = MagicMock(side_effect=[
        ValidationResult(accepted=False, issues=["Scope mismatch: Answer mentions ISO 14001"]),
        ValidationResult(accepted=True, issues=[])
    ])
    
    # Execute
    with patch("app.agent.application.classify_intent", return_value=QueryIntent(mode="explicativa")), \
         patch("app.agent.application.build_retrieval_plan", return_value=RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2, requested_standards=("ISO 9001",))):
        
        result = await use_case.execute(cmd)
        
    # Assertions
    assert result.validation.accepted is True
    assert "attempts" in result.retrieval.trace
    attempts = result.retrieval.trace["attempts"]
    assert len(attempts) == 2
    assert attempts[1]["action"] == "filter_evidence_to_scope_and_regenerate"
    
    # Verify generator call 2 had instruction
    call2_args = mock_generator.generate.call_args_list[1]
    assert "[INSTRUCCION INTERNA]" in call2_args.kwargs["query"]
    assert "ISO 9001" in call2_args.kwargs["query"]
    
    # Verify chunks passed to attempt 2 were filtered (only C1)
    chunks_att2 = call2_args.kwargs["chunks"]
    assert len(chunks_att2) == 1
    assert chunks_att2[0].source == "C1"

@pytest.mark.asyncio
async def test_silent_correction_literal_mismatch_boosts_recall(use_case, mock_retriever, mock_generator, mock_validator):
    # Setup
    cmd = HandleQuestionCommand(query="literal question", tenant_id="t1", collection_id="c1", scope_label="s1")
    
    mock_retriever.retrieve_chunks = AsyncMock(return_value=[])
    mock_retriever.retrieve_summaries = AsyncMock(return_value=[])
    
    mock_generator.generate = AsyncMock(return_value=AnswerDraft(text="draft", mode="literal_lista"))
    
    mock_validator.validate = MagicMock(side_effect=[
        ValidationResult(accepted=False, issues=["Literal clause mismatch"]),
        ValidationResult(accepted=True, issues=[])
    ])
    
    # Execute
    with patch("app.agent.application.classify_intent", return_value=QueryIntent(mode="literal_lista")), \
         patch("app.agent.application.build_retrieval_plan", return_value=RetrievalPlan(mode="literal_lista", chunk_k=5, chunk_fetch_k=20, summary_k=2)):
        
        result = await use_case.execute(cmd)
        
    # Assertions
    assert result.validation.accepted is True
    attempts = result.retrieval.trace["attempts"]
    assert len(attempts) == 2
    assert attempts[1]["action"] == "increase_fetch_k_and_reretrieve"
    
    # Verify retrieval called twice, second time with boosted fetch_k
    assert mock_retriever.retrieve_chunks.call_count == 2
    plan2 = mock_retriever.retrieve_chunks.call_args_list[1].kwargs["plan"]
    assert plan2.chunk_fetch_k >= 280
    assert plan2.chunk_k >= 55

@pytest.mark.asyncio
async def test_silent_correction_missing_citations_regenerates_strict(use_case, mock_retriever, mock_generator, mock_validator):
    cmd = HandleQuestionCommand(query="q", tenant_id="t", collection_id="c", scope_label="s")
    
    mock_retriever.retrieve_chunks = AsyncMock(return_value=[])
    mock_retriever.retrieve_summaries = AsyncMock(return_value=[])
    mock_generator.generate = AsyncMock(return_value=AnswerDraft(text="draft", mode="explicativa"))
    
    mock_validator.validate = MagicMock(side_effect=[
        ValidationResult(accepted=False, issues=["Missing explicit source markers"]),
        ValidationResult(accepted=True, issues=[])
    ])
    
    with patch("app.agent.application.classify_intent", return_value=QueryIntent(mode="explicativa")), \
         patch("app.agent.application.build_retrieval_plan", return_value=RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2)):
         
        result = await use_case.execute(cmd)
        
    assert result.validation.accepted is True
    assert result.retrieval.trace["attempts"][1]["action"] == "regenerate_with_strict_citation_instruction"
    
    # Verify instruction in attempt 2
    call2 = mock_generator.generate.call_args_list[1]
    assert "marcadores C#/R#" in call2.kwargs["query"]

@pytest.mark.asyncio
async def test_silent_correction_persistent_failure(use_case, mock_retriever, mock_generator, mock_validator):
    cmd = HandleQuestionCommand(query="q", tenant_id="t", collection_id="c", scope_label="s")
    
    # Needs valid evidence to pass the filter check in loop
    chunks = [EvidenceItem(source="C1", content="content", metadata={"row": {"source_standard": "ISO 9001"}})]
    mock_retriever.retrieve_chunks = AsyncMock(return_value=chunks)
    mock_retriever.retrieve_summaries = AsyncMock(return_value=[])
    mock_generator.generate = AsyncMock(return_value=AnswerDraft(text="draft", mode="explicativa", evidence=chunks))
    
    # Fails both times
    mock_validator.validate = MagicMock(return_value=ValidationResult(accepted=False, issues=["Scope mismatch: answer mentions incorrect standard", "Persistence"]))
    
    with patch("app.agent.application.classify_intent", return_value=QueryIntent(mode="explicativa")), \
         patch("app.agent.application.build_retrieval_plan", return_value=RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2, requested_standards=("ISO 9001",))):
         
        # Mock detect_scope_candidates to return < 2 scopes so we don't bail to clarification immediately
        with patch("app.agent.application.detect_scope_candidates", return_value=["ISO 9001"]):
             # Trigger scope mismatch logic (Task F case 1) but ensuring it fails again
             result = await use_case.execute(cmd)
    
    # Should perform 2 attempts but ultimately fail (or return the blocked answer if logic dictates)
    attempts = result.retrieval.trace.get("attempts", [])
    assert len(attempts) == 2
    assert result.validation.accepted is False or "⚠️" in result.answer.text # Depends on fallback logic
    # In application.py, if validation fails scope check after retry, it returns a blockage or clarification.
    # Since we mocked standard scope mismatch logic, we expect it to hit the blockage return at the end.
    assert "Respuesta bloqueada" in result.answer.text
