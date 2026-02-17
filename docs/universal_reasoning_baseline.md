# Universal Reasoning Baseline (Fase 0)

Fecha de congelamiento: 2026-02-17

## Benchmark base

- Dataset de referencia ISO: `tests/evaluation/iso_auditor_benchmark.json`.
- Este benchmark permanece como baseline de no-regresión para la migración de motor.

## Deuda de pruebas existente (no bloqueante)

- `tests/unit/test_usecase_hitl_mode.py`
- `tests/unit/test_retrieval_planner_profile_patterns.py`
- `tests/unit/test_literal_evidence_validator.py`

Estas pruebas representan deuda previa al `universal_flow` y se mantienen fuera del scope de estabilización inicial.
