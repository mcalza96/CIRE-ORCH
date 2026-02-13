from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _EndpointMetrics:
    requests_total: int = 0
    successes_total: int = 0
    failures_total: int = 0
    fallback_retries_total: int = 0
    degraded_responses_total: int = 0


class RetrievalMetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics: dict[str, _EndpointMetrics] = defaultdict(_EndpointMetrics)

    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self._metrics[endpoint].requests_total += 1

    def record_success(self, endpoint: str) -> None:
        with self._lock:
            self._metrics[endpoint].successes_total += 1

    def record_failure(self, endpoint: str) -> None:
        with self._lock:
            self._metrics[endpoint].failures_total += 1

    def record_fallback_retry(self, endpoint: str) -> None:
        with self._lock:
            self._metrics[endpoint].fallback_retries_total += 1

    def record_degraded_response(self, endpoint: str) -> None:
        with self._lock:
            self._metrics[endpoint].degraded_responses_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoints": {
                    key: {
                        "requests_total": value.requests_total,
                        "successes_total": value.successes_total,
                        "failures_total": value.failures_total,
                        "fallback_retries_total": value.fallback_retries_total,
                        "degraded_responses_total": value.degraded_responses_total,
                    }
                    for key, value in self._metrics.items()
                }
            }


retrieval_metrics_store = RetrievalMetricsStore()
