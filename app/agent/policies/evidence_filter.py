
from typing import Any

from app.agent.models import EvidenceItem, RetrievalPlan

class EvidenceFilter:
    def filter_by_standards(
        self, 
        evidence: list[EvidenceItem], 
        requested_standards: tuple[str, ...]
    ) -> list[EvidenceItem]:
        if not requested_standards:
            return evidence
            
        allowed_upper = {s.upper() for s in requested_standards if s}
        out: list[EvidenceItem] = []
        
        for item in evidence:
            row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
            # If no metadata row, keep it (might be a generated summary or special item)
            if not isinstance(row, dict):
                out.append(item)
                continue
                
            std = self._extract_row_standard(row)
            # If we can't determine standard, keep it safe
            if not std:
                out.append(item)
                continue
                
            # Check if any allowed standard is substring of row's standard
            if any(target in std for target in allowed_upper):
                out.append(item)
                
        return out

    def _extract_row_standard(self, row: dict[str, Any]) -> str:
        meta_raw = row.get("metadata")
        meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
        
        candidates = [
            meta.get("source_standard"),
            meta.get("standard"),
            meta.get("scope"),
            row.get("source_standard"),
        ]
        
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        return ""
