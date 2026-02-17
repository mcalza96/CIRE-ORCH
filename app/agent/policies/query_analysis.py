
import re

def is_literal_query(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "textualmente",
        "literal",
        "verbatim",
        "transcribe",
        "cita",
        "citas",
        "que exige",
        "qué exige",
    )
    if any(token in lowered for token in markers):
        return True
    return bool(re.search(r"\bcl(?:a|á)usula\s*\d+(?:\.\d+)+\b", lowered))


def is_list_literal_query(text: str) -> bool:
    lowered = (text or "").lower()
    markers = ("enumera", "lista", "listado", "viñetas", "vinetas")
    return any(token in lowered for token in markers)


def literal_force_eligible(text: str) -> bool:
    query = str(text or "").strip()
    if not is_literal_query(query):
        return False
    if len(query) > 220:
        return False
    scopes = re.findall(r"\biso\s*[-:]?\s*\d{4,5}\b", query, flags=re.IGNORECASE)
    if len(scopes) >= 2:
        return False
    if query.count("?") >= 2:
        return False
    return True
