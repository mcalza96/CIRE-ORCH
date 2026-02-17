
from typing import Any

from app.agent.models import RetrievalPlan, EvidenceItem
from app.cartridges.models import AgentProfile
from app.agent.components.parsing import extract_row_standard

class ScopePolicy:
    def detect_missing_scopes(
        self, 
        evidence: list[EvidenceItem], 
        plan: RetrievalPlan
    ) -> list[str]:
        """
        Identify requested standards that are not present in the evidence.
        """
        if len(plan.requested_standards) < 2:
            return []
            
        required = {s.upper() for s in plan.requested_standards if s}
        found: set[str] = set()
        
        for item in evidence:
            std = extract_row_standard(item)
            if not std:
                continue
            for req in required:
                # Flexible matching: ISO 9001 in ISO 9001:2015
                if req in std or std in req:
                    found.add(req)
                    
        return sorted(list(required - found))

    def evaluate_coverage(
        self,
        evidence: list[EvidenceItem],
        plan: RetrievalPlan
    ) -> dict[str, Any]:
        """
        Evaluate if the retrieval covers all requested scopes.
        """
        missing = self.detect_missing_scopes(evidence, plan)
        return {
            "missing_scopes": missing,
            "coverage_complete": len(missing) == 0,
            "scope_mismatch": bool(missing)
        }
