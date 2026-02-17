
from app.agent.policies.query_splitter import QuerySplitter, QuerySplitterConfig
from app.agent.policies.scope_policy import ScopePolicy
from app.agent.policies.retry_policy import RetryPolicy
from app.agent.models import RetrievalPlan, EvidenceItem, QueryIntent
from dataclasses import dataclass

def test_query_splitter_basic():
    cfg = QuerySplitterConfig(min_length=2)
    splitter = QuerySplitter(cfg)
    parts = splitter.split("Hola? Como estas")
    assert len(parts) == 2
    assert "Hola" in parts
    assert "Como estas" in parts

def test_query_splitter_single():
    splitter = QuerySplitter()
    parts = splitter.split("Hola mundo")
    assert len(parts) == 1
    assert parts[0] == "Hola mundo"

def test_scope_policy_missing():
    policy = ScopePolicy()
    plan = RetrievalPlan(
        mode="literal_normativa", 
        chunk_k=1, chunk_fetch_k=1, summary_k=1, 
        require_literal_evidence=True,
        requested_standards=("ISO 9001", "ISO 45001")
    )
    
    # Mock evidence with metadata
    @dataclass
    class MockEvidence:
        metadata: dict
        content: str = ""
        
    evidence = [
        MockEvidence(metadata={"row": {"source_standard": "ISO 9001:2015"}})
    ]
    
    missing = policy.detect_missing_scopes(evidence, plan)  # type: ignore
    assert "ISO 45001" in missing
    assert "ISO 9001" not in missing

def test_retry_policy_relax():
    policy = RetryPolicy()
    plan = RetrievalPlan(
        mode="literal_normativa",
        chunk_k=1, chunk_fetch_k=1, summary_k=1,
        require_literal_evidence=True,
        requested_standards=("ISO 9001", "ISO 45001")
    )
    
    next_intent = policy.determine_next_intent(plan, "scope_mismatch")
    assert next_intent is not None
    assert next_intent.mode == "comparativa"
