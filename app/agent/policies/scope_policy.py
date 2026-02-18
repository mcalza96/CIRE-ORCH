import re
from typing import Any

from app.agent.models import RetrievalPlan, EvidenceItem
from app.agent.components.parsing import extract_row_standard


_SCOPE_NOISE_PATTERN = re.compile(r"[^A-Z0-9]+")
_SCOPE_YEAR_SUFFIX_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _normalize_scope(value: str) -> str:
    cleaned = str(value or "").strip().upper()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("/", " ")
    cleaned = cleaned.replace(":", " ")
    cleaned = _SCOPE_NOISE_PATTERN.sub(" ", cleaned)
    cleaned = _SCOPE_YEAR_SUFFIX_PATTERN.sub("", cleaned)
    return " ".join(part for part in cleaned.split() if part)


def _scope_key(value: str) -> str:
    normalized = _normalize_scope(value)
    if not normalized:
        return ""
    parts = normalized.split()
    first_numeric = next((part for part in parts if part.isdigit()), "")
    if first_numeric and len(parts) >= 1:
        if parts[0] == "ISO" and len(parts) >= 3 and parts[1] == "IEC":
            return f"ISO IEC {first_numeric}"
        return f"{parts[0]} {first_numeric}"
    return normalized


class ScopePolicy:
    def detect_missing_scopes(self, evidence: list[EvidenceItem], plan: RetrievalPlan) -> list[str]:
        """
        Identify requested standards that are not present in the evidence.
        """
        if len(plan.requested_standards) < 2:
            return []

        required_by_key: dict[str, str] = {}
        for scope in plan.requested_standards:
            key = _scope_key(scope)
            if key and key not in required_by_key:
                required_by_key[key] = scope.upper()

        if not required_by_key:
            return []

        found_keys: set[str] = set()
        for item in evidence:
            key = _scope_key(extract_row_standard(item))
            if key and key in required_by_key:
                found_keys.add(key)

        missing = [required_by_key[key] for key in required_by_key if key not in found_keys]
        return sorted(missing)

    def evaluate_coverage(
        self, evidence: list[EvidenceItem], plan: RetrievalPlan
    ) -> dict[str, Any]:
        """
        Evaluate if the retrieval covers all requested scopes.
        """
        missing = self.detect_missing_scopes(evidence, plan)
        return {
            "missing_scopes": missing,
            "coverage_complete": len(missing) == 0,
            "scope_mismatch": bool(missing),
        }
