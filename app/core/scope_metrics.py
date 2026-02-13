from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _TenantScopeMetrics:
    requests_total: int = 0
    scope_clarification_required: int = 0
    scope_mismatch_detected: int = 0
    scope_mismatch_blocked: int = 0


class ScopeMetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics: dict[str, _TenantScopeMetrics] = defaultdict(_TenantScopeMetrics)

    @staticmethod
    def _tenant(tenant_id: str | None) -> str:
        return str(tenant_id or "unknown")

    def record_request(self, tenant_id: str | None) -> None:
        with self._lock:
            self._metrics[self._tenant(tenant_id)].requests_total += 1

    def record_clarification(self, tenant_id: str | None) -> None:
        with self._lock:
            self._metrics[self._tenant(tenant_id)].scope_clarification_required += 1

    def record_mismatch_detected(self, tenant_id: str | None) -> None:
        with self._lock:
            self._metrics[self._tenant(tenant_id)].scope_mismatch_detected += 1

    def record_mismatch_blocked(self, tenant_id: str | None) -> None:
        with self._lock:
            self._metrics[self._tenant(tenant_id)].scope_mismatch_blocked += 1

    def snapshot(self, tenant_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if tenant_id:
                key = self._tenant(tenant_id)
                item = self._metrics.get(key, _TenantScopeMetrics())
                return {"tenant_id": key, **self._serialize(item)}

            return {
                "tenants": {
                    key: self._serialize(value)
                    for key, value in self._metrics.items()
                }
            }

    @staticmethod
    def _serialize(item: _TenantScopeMetrics) -> dict[str, Any]:
        mismatch_block_ratio = 0.0
        if item.scope_mismatch_detected > 0:
            mismatch_block_ratio = round(item.scope_mismatch_blocked / item.scope_mismatch_detected, 4)

        clarification_ratio = 0.0
        if item.requests_total > 0:
            clarification_ratio = round(item.scope_clarification_required / item.requests_total, 4)

        return {
            "requests_total": item.requests_total,
            "scope_clarification_required": item.scope_clarification_required,
            "scope_mismatch_detected": item.scope_mismatch_detected,
            "scope_mismatch_blocked": item.scope_mismatch_blocked,
            "scope_mismatch_block_ratio": mismatch_block_ratio,
            "scope_clarification_ratio": clarification_ratio,
        }


scope_metrics_store = ScopeMetricsStore()
