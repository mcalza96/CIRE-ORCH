from app.agent.components.citations import build_citation_bundle
from app.agent.models import EvidenceItem
from app.cartridges.models import AgentProfile


def test_citation_bundle_extracts_clause_id_from_content_marker() -> None:
    profile = AgentProfile(profile_id="p")
    evidence = [
        EvidenceItem(
            source="C1",
            content="[CLAUSE_ID: 5.3] texto de evidencia",
            score=0.8,
            metadata={
                "row": {
                    "metadata": {
                        "source_standard": "ISO 9001",
                    }
                }
            },
        )
    ]
    citations, details, quality = build_citation_bundle(
        answer_text="segun C1",
        evidence=evidence,
        profile=profile,
    )
    assert citations == ["C1"]
    assert details[0]["clause"] == "5.3"
    assert details[0]["missing_fields"] == []
    assert quality["structured_count"] == 1


def test_citation_bundle_marks_noise_items() -> None:
    profile = AgentProfile(profile_id="p")
    evidence = [
        EvidenceItem(
            source="C2",
            content="Traducción oficial índice y prólogo",
            score=0.5,
            metadata={"row": {"metadata": {}}},
        )
    ]
    citations, details, quality = build_citation_bundle(
        answer_text="",
        evidence=evidence,
        profile=profile,
    )
    assert citations == []
    assert details[0]["noise"] is True
    assert quality["discarded_noise"] == 1


def test_citation_bundle_reports_missing_scope_citations() -> None:
    profile = AgentProfile(profile_id="p")
    evidence = [
        EvidenceItem(
            source="C1",
            content="[CLAUSE_ID: 5.1] evidencia calidad",
            score=0.9,
            metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
        )
    ]
    _, _, quality = build_citation_bundle(
        answer_text="segun C1",
        evidence=evidence,
        profile=profile,
        requested_scopes=("ISO 9001", "ISO 14001", "ISO 45001"),
    )
    assert quality["citations_per_scope"].get("ISO 9001") == 1
    assert "ISO 14001" in quality["missing_scope_citations"]
    assert "ISO 45001" in quality["missing_scope_citations"]


def test_citation_bundle_trusts_rag_noise_flags() -> None:
    profile = AgentProfile(profile_id="p")
    evidence = [
        EvidenceItem(
            source="C9",
            content="Texto normativo legitimo sin palabra indice",
            score=0.7,
            metadata={"row": {"metadata": {"is_toc": True, "source_standard": "ISO 9001"}}},
        )
    ]
    citations, details, quality = build_citation_bundle(
        answer_text="",
        evidence=evidence,
        profile=profile,
    )
    assert citations == []
    assert details[0]["noise"] is True
    assert quality["discarded_noise"] == 1
