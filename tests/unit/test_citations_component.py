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
