from __future__ import annotations

import re

from app.agent.models import QueryIntent, RetrievalPlan


LITERAL_LIST_HINTS = (
    "entradas",
    "salidas",
    "lista",
    "exclusivas",
    "enumera",
    "listado",
    "viñetas",
)

LITERAL_NORMATIVE_HINTS = (
    "clausula",
    "cláusula",
    "documento obligatorio",
    "obligatorio",
    "exacto",
    "literal",
    "que exige",
    "qué exige",
)

COMPARATIVE_HINTS = ("compar", "difer", "vs", "ambas", "respecto")

# Markers that suggest the user wants interpretation/impacts, not literal extraction.
INTERPRETIVE_HINTS = (
    "obliga ",
    "implica",
    "impacto",
    "interaccion",
    "interacción",
    "reevalu",
    "re-evalu",
    "ciclo de vida",
    "perspectiva",
    "gestion del cambio",
    "gestión del cambio",
    "validacion",
    "validación",
)

SCOPE_HINTS: dict[str, tuple[str, ...]] = {
    "ISO 9001": ("calidad", "cliente", "producto", "servicio"),
    "ISO 14001": ("ambient", "legal", "cumplimiento", "aspecto ambiental"),
    "ISO 45001": ("seguridad", "salud", "sst", "riesgo laboral", "trabajador"),
}

CONFLICT_MARKERS = (
    "conflicto",
    "represalia",
    "confidencial",
    "denuncia",
    "anonim",
    "proteccion de datos",
    "protección de datos",
    "rrhh",
    "se niega",
)

EVIDENCE_MARKERS = (
    "evidencia",
    "trazabilidad",
    "verificar",
    "registros",
    "informacion documentada",
    "información documentada",
)


def extract_requested_standards(query: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    text = query or ""
    for match in re.findall(r"\biso\s*[-:]?\s*(\d{4,5})\b", text, flags=re.IGNORECASE):
        value = f"ISO {match}"
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    for match in re.findall(r"\b(9001|14001|45001)\b", text):
        value = f"ISO {match}"
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return tuple(ordered)


def has_clause_reference(query: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)+\b", (query or "")))


def suggest_scope_candidates(query: str) -> tuple[str, ...]:
    text = (query or "").strip().lower()
    ranked: list[str] = []
    for standard, hints in SCOPE_HINTS.items():
        if any(h in text for h in hints):
            ranked.append(standard)

    if ranked:
        seen: set[str] = set()
        ordered = [s for s in ranked if not (s in seen or seen.add(s))]
        return tuple(ordered)

    return ("ISO 9001", "ISO 14001", "ISO 45001")


def detect_scope_candidates(query: str) -> tuple[str, ...]:
    requested = list(extract_requested_standards(query))
    text = (query or "").strip().lower()
    for standard, hints in SCOPE_HINTS.items():
        if standard in requested:
            continue
        if any(h in text for h in hints):
            requested.append(standard)
    return tuple(requested)


def detect_conflict_objectives(query: str) -> bool:
    text = (query or "").strip().lower()
    has_conflict = any(marker in text for marker in CONFLICT_MARKERS)
    has_evidence = any(marker in text for marker in EVIDENCE_MARKERS)
    return has_conflict and has_evidence


def classify_intent(query: str) -> QueryIntent:
    text = (query or "").strip().lower()
    requested_standards = extract_requested_standards(query)
    # Multi-ISO questions that ask for cross-impact between clauses behave better as "comparativa"
    # than as a single-standard literal extraction.
    if len(requested_standards) >= 2 and any(k in text for k in INTERPRETIVE_HINTS):
        return QueryIntent(mode="comparativa", rationale="multi-standard interpretive cross-impact")
    if any(h in text for h in LITERAL_LIST_HINTS):
        return QueryIntent(mode="literal_lista", rationale="list-like normative query")
    if any(h in text for h in LITERAL_NORMATIVE_HINTS):
        # If the user is asking "how/impact" even while referencing clauses, treat as interpretive.
        # But if it's a specific clause reference without interpretive intent, it should be literal.
        if any(h in text for h in INTERPRETIVE_HINTS):
            if len(requested_standards) >= 2:
                return QueryIntent(mode="comparativa", rationale="interpretive question with clause refs")
            return QueryIntent(mode="explicativa", rationale="interpretive question with clause refs")
        
        if has_clause_reference(query) and not requested_standards:
            return QueryIntent(mode="ambigua_scope", rationale="clause reference without explicit standard scope")
            
        return QueryIntent(mode="literal_normativa", rationale="normative exactness query")
    if any(h in text for h in COMPARATIVE_HINTS):
        return QueryIntent(mode="comparativa", rationale="cross-standard comparison")
    return QueryIntent(mode="explicativa", rationale="general explanatory query")


def build_retrieval_plan(intent: QueryIntent, query: str = "") -> RetrievalPlan:
    requested_standards = extract_requested_standards(query)
    if intent.mode in {"literal_lista", "literal_normativa"}:
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=45,
            chunk_fetch_k=220,
            summary_k=3,
            require_literal_evidence=True,
            requested_standards=requested_standards,
        )
    if intent.mode == "comparativa":
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=35,
            chunk_fetch_k=140,
            summary_k=5,
            # Comparativa is typically interpretive; still grounded in retrieved context,
            # but do not force clause-by-clause literal quoting.
            require_literal_evidence=False,
            requested_standards=requested_standards,
        )
    if intent.mode == "ambigua_scope":
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=0,
            chunk_fetch_k=0,
            summary_k=0,
            require_literal_evidence=True,
            requested_standards=requested_standards,
        )
    return RetrievalPlan(
        mode=intent.mode,
        chunk_k=30,
        chunk_fetch_k=120,
        summary_k=5,
        require_literal_evidence=False,
        requested_standards=requested_standards,
    )
