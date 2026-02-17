# ISO Auditor Benchmark (P2)

Dataset y runner para calibrar `min_score` + `search_hints` de `iso_auditor`.

## Archivos
- `iso_auditor_benchmark.json`: 120 casos (`40 literal`, `40 comparativa`, `40 explicativa`).
- `run_iso_auditor_benchmark.py`: ejecuta benchmark contra `POST /api/v1/knowledge/answer`.

## Uso rapido
```bash
cd /Users/mcalzadilla/cire/orch
export ORCH_BENCH_TENANT_ID="<tenant_uuid>"
export ORCH_BENCH_COLLECTION_ID="<collection_uuid>"
export AUTH_BEARER_TOKEN="<jwt_opcional>"

./tests/evaluation/run_iso_auditor_benchmark.py \
  --variant-label baseline \
  --base-url http://localhost:8001 \
  --fail-on-thresholds
```

## Matriz sugerida (P2)
Ejecutar una corrida por variante y comparar reportes:
1. `min_score=0.72` + hints baseline
2. `min_score=0.70` + hints baseline
3. `min_score=0.68` + hints baseline
4. `min_score=0.65` + hints baseline
5. variantes equivalentes agregando hints expandidos

## KPIs de promocion
- `citation_marker_rate >= 98`
- `standard_coverage_rate >= 90`
- `literal_mode_retention_rate >= 95`
- revisar `false_positive_partial_rate` y `latency_p95_ms` vs baseline
