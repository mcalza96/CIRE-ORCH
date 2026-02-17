from __future__ import annotations

import re


def ensure_citation_footer(text: str, references: list[str]) -> str:
    output = str(text or "").strip()
    if not output:
        return output
    if re.search(r"\b[CR]\d+\b", output):
        return output
    refs = [str(ref).strip() for ref in references if str(ref).strip()]
    if not refs:
        return output
    return output.rstrip() + "\n\nReferencias revisadas: " + ", ".join(refs)
