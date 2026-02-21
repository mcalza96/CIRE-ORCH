from __future__ import annotations

import argparse
from dataclasses import dataclass


BATCH_TERMINAL_STATES = {
    "done",
    "error",
    "cancelled",
    "completed",
    "failed",
}

JOB_TERMINAL_STATES = {"completed", "failed", "cancelled", "error"}
HEARTBEAT_SECONDS = 1.5

_BATCH_STAGE_LABELS: dict[str, str] = {
    "RECEIVED": "Analizando batch de ingesta",
    "VALIDATING": "Validando archivos y metadatos",
    "UPLOADING": "Subiendo archivos",
    "PARSING": "Extrayendo contenido",
    "CHUNKING": "Fragmentando contenido",
    "EMBEDDING": "Generando embeddings",
    "PERSISTING": "Persistiendo chunks indexables",
    "ENRICHING": "Ejecutando enriquecimientos",
    "GRAPH": "Construyendo grafo semantico",
    "RAPTOR": "Construyendo jerarquia RAPTOR",
    "FINALIZING": "Finalizando batch",
    "OTHER": "Procesando pipeline",
}

_JOB_STAGE_LABELS: dict[str, str] = {
    "pending": "En cola para ejecutar enrichment",
    "processing": "Ejecutando enrichment",
    "completed": "Enrichment finalizado",
    "failed": "Enrichment finalizado con error",
    "cancelled": "Enrichment cancelado",
    "error": "Enrichment finalizado con error",
}


@dataclass
class PollStep:
    heartbeat_label: str
    event_emitted: bool = False
    terminal: bool = False
    terminal_message: str = ""


@dataclass
class IngestionRuntime:
    args: argparse.Namespace
    operation: str
    access_token: str
    tenant_id: str
