from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScopeValidationError(Exception):
    message: str
    violations: list[dict[str, Any]]
    warnings: list[dict[str, Any]] | None = None
    normalized_scope: dict[str, Any] | None = None
    query_scope: dict[str, Any] | None = None

