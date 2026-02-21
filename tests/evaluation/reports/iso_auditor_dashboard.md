# ISO Auditor Dashboard

- Latest report: `tests/evaluation/reports/iso_auditor_benchmark_semantic-benchmark-calibrated_20260217T214749Z.json`
- Best coverage: `phase2-guided-retry-tuned-r2` (81.82)
- Best literal retention: `phase2-force-literal-retry` (80.00)

## Latest Metrics

- citation=100.00 | citation_suff=100.00 | coverage=72.73 | semantic=90.91 | clause=90.91
- hallucination_guard=100.00 | literal_obedience=20.00 | literal_mode_retention=20.00 | answerable=100.00 | false_positive=0.00 | p95_ms=3379.03

## Recent Runs

| variant | generated_at | citation | citation_suff | coverage | semantic | clause | hallucination_guard | literal_obedience | literal | answerable | false_positive | p95_ms | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic-benchmark-calibrated | 2026-02-17T21:47:49.656312+00:00 | 100.00 | 100.00 | 72.73 | 90.91 | 90.91 | 100.00 | 20.00 | 20.00 | 100.00 | 0.00 | 3379.03 | OK |
| semantic-benchmark-v1-r2 | 2026-02-17T21:38:52.772318+00:00 | 100.00 | 100.00 | 54.55 | 81.82 | 81.82 | 100.00 | 20.00 | 20.00 | 100.00 | 100.00 | 3930.27 | FAIL |
| semantic-benchmark-v1 | 2026-02-17T21:36:19.325220+00:00 | 100.00 | 100.00 | 63.64 | 81.82 | 100.00 | 9.09 | 20.00 | 20.00 | 9.09 | 50.00 | 4065.09 | FAIL |
| phase2-graph-gaps | 2026-02-17T21:19:40.547424+00:00 | 100.00 | 0.00 | 63.64 | 0.00 | 0.00 | 0.00 | 0.00 | 20.00 | 100.00 | 50.00 | 4460.00 | FAIL |
| phase5-post-reingest | 2026-02-17T19:33:24.121301+00:00 | 100.00 | 0.00 | 63.64 | 0.00 | 0.00 | 0.00 | 0.00 | 20.00 | 100.00 | 50.00 | 4382.99 | FAIL |
| phase2-literal-eligibility | 2026-02-17T19:26:15.151659+00:00 | 100.00 | 0.00 | 72.73 | 0.00 | 0.00 | 0.00 | 0.00 | 20.00 | 100.00 | 50.00 | 6328.94 | FAIL |
| phase2-force-literal-retry | 2026-02-17T19:23:00.962087+00:00 | 100.00 | 0.00 | 63.64 | 0.00 | 0.00 | 0.00 | 0.00 | 80.00 | 81.82 | 100.00 | 3752.28 | FAIL |
| phase2-guided-retry-tuned-r2 | 2026-02-17T19:19:11.761391+00:00 | 100.00 | 0.00 | 81.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 50.00 | 5058.37 | FAIL |
| phase2-guided-retry-tuned | 2026-02-17T18:51:29.273138+00:00 | 72.73 | 0.00 | 63.64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 72.73 | 0.00 | 4333.31 | FAIL |
| phase2-literal-guided-retry | 2026-02-17T18:46:39.600659+00:00 | 100.00 | 0.00 | 54.55 | 0.00 | 0.00 | 0.00 | 0.00 | 80.00 | 81.82 | 100.00 | 8875.45 | FAIL |
| doc-hints-update-fixed-r2 | 2026-02-17T18:35:00.934587+00:00 | 100.00 | 0.00 | 81.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 5816.93 | FAIL |
| doc-hints-update-fixed | 2026-02-17T18:31:29.912331+00:00 | 90.91 | 0.00 | 54.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 90.91 | 50.00 | 5342.65 | FAIL |
