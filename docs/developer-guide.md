# Developer Guide

## Flujo recomendado

1. Cambios pequeños y enfocados.
2. Tests unitarios del orquestador en el mismo PR.
3. Documentación actualizada cuando cambie el contrato HTTP o políticas.

## Quality gates

```bash
ruff check runtime tests chat_cli.py
python -m compileall runtime tests chat_cli.py
pytest tests/unit -q
```

## Tests vigentes

- `tests/unit/test_qa_orchestrator_policies.py`
- `tests/unit/test_qa_orchestrator_use_case.py`
- `tests/unit/test_literal_evidence_validator.py`

## Dependencias

- `requirements-core.txt` es el set mínimo para runtime local.
- `requirements.txt` replica pins para entornos que no usan split de requirements.
