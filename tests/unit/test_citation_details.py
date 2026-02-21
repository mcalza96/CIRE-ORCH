from app.agent.types.models import EvidenceItem
from app.api.v1.routes.knowledge import _build_citation_details


def test_build_citation_details_includes_metadata_and_snippet():
    evidence = [
        EvidenceItem(
            source="C4",
            content="ISO 9001 exige controlar la informacion documentada y su disponibilidad.",
            score=0.77,
            metadata={
                "row": {
                    "metadata": {
                        "source_standard": "ISO 9001",
                        "clause_id": "7.5.3",
                    }
                }
            },
        )
    ]

    out = _build_citation_details("Hallazgo clave (C4)", evidence)
    assert len(out) == 1
    assert out[0]["id"] == "C4"
    assert out[0]["standard"] == "ISO 9001"
    assert out[0]["clause"] == "7.5.3"
    assert out[0]["used_in_answer"] is True
    assert "informacion documentada" in out[0]["snippet"].lower()


def test_build_citation_details_prioritizes_used_markers_first():
    evidence = [
        EvidenceItem(source="C2", content="segundo", metadata={}),
        EvidenceItem(source="C1", content="primero", metadata={}),
    ]
    out = _build_citation_details("C1", evidence)
    assert [row["id"] for row in out] == ["C1", "C2"]
    assert out[0]["used_in_answer"] is True
    assert out[1]["used_in_answer"] is False
