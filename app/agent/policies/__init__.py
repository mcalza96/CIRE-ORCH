from __future__ import annotations

import re
from typing import Any

from app.cartridges.models import AgentProfile
from app.agent.models import QueryIntent, RetrievalPlan


LITERAL_LIST_HINTS = ("lista", "enumera", "listado", "vinetas")

LITERAL_NORMATIVE_HINTS = (
    "texto exacto",
    "literal",
    "que exige",
    "requisito",
    "obligatorio",
)

COMPARATIVE_HINTS = ("compar", "difer", "vs", "entre", "respecto")

# Markers that suggest the user wants interpretation/impacts, not literal extraction.
INTERPRETIVE_HINTS = (
    "implica",
    "impacto",
    "analiza",
    "analice",
    "causa",
    "por que",
    "relaciona",
)

SCOPE_HINTS: dict[str, tuple[str, ...]] = {}

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


def _has_any_keyword(text: str, values: tuple[str, ...]) -> bool:
    hay = (text or "").lower()
    return any(str(value).strip().lower() in hay for value in values if str(value).strip())


def _has_all_keywords(text: str, values: tuple[str, ...]) -> bool:
    hay = (text or "").lower()
    checks = [str(value).strip().lower() for value in values if str(value).strip()]
    if not checks:
        return True
    return all(value in hay for value in checks)


def _has_any_pattern(text: str, values: tuple[str, ...]) -> bool:
    for value in values:
        pattern = str(value).strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _has_all_patterns(text: str, values: tuple[str, ...]) -> bool:
    checks = [str(value).strip() for value in values if str(value).strip()]
    if not checks:
        return True
    for pattern in checks:
        try:
            if not re.search(pattern, text, flags=re.IGNORECASE):
                return False
        except re.error:
            return False
    return True


def _classify_with_profile_rules(
    query: str,
    profile: AgentProfile,
) -> tuple[QueryIntent, dict[str, Any]] | None:
    modes = profile.query_modes.modes
    rules = profile.query_modes.intent_rules
    if not modes or not rules:
        return None

    text = str(query or "")
    lowered = text.lower()
    requested_standards = extract_requested_scopes(query, profile=profile)
    for rule in rules:
        mode = str(rule.mode or "").strip()
        if not mode or mode not in modes:
            continue
        if not _has_all_keywords(lowered, tuple(rule.all_keywords)):
            continue
        if not _has_all_patterns(text, tuple(rule.all_patterns)):
            continue
        if not _has_all_keywords(lowered, tuple(rule.all_markers)):
            continue

        any_kw = tuple(rule.any_keywords)
        any_pat = tuple(rule.any_patterns)
        any_mk = tuple(rule.any_markers)

        has_any_constraints = bool(any_kw or any_pat or any_mk)
        if has_any_constraints:
            matched_any = False
            if any_kw and _has_any_keyword(lowered, any_kw):
                matched_any = True
            if any_pat and _has_any_pattern(text, any_pat):
                matched_any = True
            if any_mk and _has_any_keyword(lowered, any_mk):
                matched_any = True
            
            if not matched_any:
                continue

        return QueryIntent(mode=mode, rationale=f"profile_rule:{rule.id}"), {
            "version": "profile_rules_v1",
            "mode": mode,
            "confidence": 0.85,
            "reasons": [f"rule:{rule.id}"],
            "features": {
                "requested_scopes_count": len(requested_standards),
            },
            "blocked_modes": [],
        }

    default_mode = str(profile.query_modes.default_mode or "").strip()
    if default_mode and default_mode in modes:
        return QueryIntent(mode=default_mode, rationale="profile_default_mode"), {
            "version": "profile_rules_v1",
            "mode": default_mode,
            "confidence": 0.55,
            "reasons": ["default_mode"],
            "features": {
                "requested_scopes_count": len(requested_standards),
            },
            "blocked_modes": [],
        }
    return None


def _router_hints(
    profile: AgentProfile | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    tuple[str, ...],
]:
    if profile is None:
        return (
            LITERAL_LIST_HINTS,
            LITERAL_NORMATIVE_HINTS,
            COMPARATIVE_HINTS,
            INTERPRETIVE_HINTS,
            SCOPE_HINTS,
            CONFLICT_MARKERS,
            EVIDENCE_MARKERS,
        )

    router = profile.router
    scope_hints: dict[str, tuple[str, ...]] = {
        key: tuple(values)
        for key, values in router.scope_hints.items()
        if isinstance(key, str) and isinstance(values, list)
    }

    return (
        tuple(router.literal_list_hints) or LITERAL_LIST_HINTS,
        tuple(router.literal_normative_hints) or LITERAL_NORMATIVE_HINTS,
        tuple(router.comparative_hints) or COMPARATIVE_HINTS,
        tuple(router.interpretive_hints) or INTERPRETIVE_HINTS,
        scope_hints or SCOPE_HINTS,
        tuple(router.conflict_markers) or CONFLICT_MARKERS,
        tuple(router.evidence_markers) or EVIDENCE_MARKERS,
    )


def _looks_like_scope_label(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"\d{2,}", text):
        return True
    if re.search(
        r"\b(?:ISO|IEC|NOM|NMX|ASTM|NFPA|OSHA|UNE|EN|IRAM|BS|DIN)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    compact = text.replace(" ", "")
    return bool(compact.isupper() and len(compact) <= 12)


def _edit_distance(a: str, b: str) -> int:
    left = str(a or "")
    right = str(b or "")
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        curr = [i]
        for j, ch_right in enumerate(right, start=1):
            cost = 0 if ch_left == ch_right else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def _canonical_scope_value(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().upper()


def _scope_catalog(profile: AgentProfile | None) -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = {}

    def _add(canonical: str, alias: str) -> None:
        can = _canonical_scope_value(canonical)
        ali = str(alias or "").strip().lower()
        if not can or not ali:
            return
        bucket = catalog.setdefault(can, set())
        bucket.add(ali)

    if profile is not None:
        for scope in profile.scope_resolution.canonical_scopes:
            _add(scope, scope)
        for canonical, aliases in profile.scope_resolution.aliases.items():
            _add(canonical, canonical)
            for alias in aliases:
                _add(canonical, alias)
        for scope, hints in profile.router.scope_hints.items():
            _add(scope, scope)
            for hint in hints:
                _add(scope, hint)

    return catalog


def _extract_digit_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for prefix, digits in re.findall(
        r"\b([A-Za-z]{2,12})\s*[-:_]?\s*(\d{2,6})\b",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(f"{prefix} {digits}")
        candidates.append(str(digits))
    for compact in re.findall(r"\b[A-Za-z]{2,12}[-_]?\d{2,6}\b", text):
        candidates.append(str(compact))
    return candidates


def _best_scope_from_token(
    token: str,
    catalog: dict[str, set[str]],
    fuzzy_enabled: bool,
) -> tuple[str, float] | None:
    text = str(token or "").strip().lower()
    if not text:
        return None
    best_scope = ""
    best_score = 0.0
    for canonical, aliases in catalog.items():
        alias_set = set(aliases)
        alias_set.add(canonical.lower())
        if text in alias_set:
            return canonical, 1.0
        if not fuzzy_enabled:
            continue
        local_best = 0.0
        for alias in alias_set:
            if not alias:
                continue
            dist = _edit_distance(text, alias)
            max_len = max(len(text), len(alias), 1)
            score = 1.0 - (dist / max_len)
            if score > local_best:
                local_best = score
        if local_best > best_score:
            best_scope = canonical
            best_score = local_best
    if best_scope:
        return best_scope, max(0.0, min(1.0, best_score))
    return None


def extract_requested_scopes(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    """Extract standard names/codes from query using profile patterns or generic fallback."""
    text = (query or "").strip()
    lower_text = text.lower()
    found: set[str] = set()
    ordered: list[str] = []

    def _add_scope(candidate: str) -> None:
        value = str(candidate or "").strip()
        if not value or value in found:
            return
        found.add(value)
        ordered.append(value)

    marker = re.search(r"__requested_scopes__=\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if marker:
        raw_list = str(marker.group(1) or "")
        for item in re.split(r"[|,;]", raw_list):
            candidate = _canonical_scope_value(item)
            if candidate:
                _add_scope(candidate)
    else:
        legacy_marker = re.search(r"__requested_scopes__=([^\s]+)", text, flags=re.IGNORECASE)
        if legacy_marker:
            legacy_raw = str(legacy_marker.group(1) or "")
            for item in re.split(r"[|,;]", legacy_raw):
                candidate = _canonical_scope_value(item)
                if candidate:
                    _add_scope(candidate)

    patterns = profile.router.scope_patterns if profile and profile.router.scope_patterns else []

    # If we have a profile with explicit patterns, use them.
    if patterns:
        for p in patterns:
            # Handle both ScopePattern objects and raw strings
            regex = p.regex if hasattr(p, "regex") else str(p)
            label = p.label if hasattr(p, "label") else None

            try:
                if re.search(regex, text, flags=re.IGNORECASE):
                    if label:
                        _add_scope(label.strip())
                    else:
                        matches = re.findall(regex, lower_text)
                        for m in matches:
                            if isinstance(m, tuple):
                                m = " ".join(filter(None, m))
                            candidate = str(m).strip().upper()
                            if _looks_like_scope_label(candidate):
                                _add_scope(candidate)
            except re.error:
                continue

    # Fallback to explicit scope hints from the active profile.
    if profile:
        for scope_label, hints in profile.router.scope_hints.items():
            if any(h.lower() in lower_text for h in hints) and scope_label not in found:
                _add_scope(scope_label.strip())

    scope_catalog = _scope_catalog(profile)
    fuzzy_enabled = bool(profile.scope_resolution.fuzzy_enabled) if profile is not None else True
    min_autoresolve = (
        float(profile.scope_resolution.min_confidence_autoresolve) if profile is not None else 0.85
    )
    min_clarify = (
        float(profile.scope_resolution.min_confidence_clarify) if profile is not None else 0.6
    )

    # Generic extraction from numeric mentions and aliases, with typo tolerance.
    token_candidates: list[str] = []
    token_candidates.extend(_extract_digit_candidates(text))
    token_candidates.extend(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", text))
    token_candidates.extend(re.findall(r"\b[\w\-]{3,}\b", text))

    for token in token_candidates:
        if not scope_catalog:
            fallback = _canonical_scope_value(token)
            if re.search(r"\d", fallback) and re.search(r"[A-Z]", fallback):
                _add_scope(fallback)
            elif re.fullmatch(r"\d{3,6}", fallback):
                _add_scope(fallback)
            continue
        match = _best_scope_from_token(token, scope_catalog, fuzzy_enabled=fuzzy_enabled)
        if not match:
            continue
        scope, score = match
        if score >= min_autoresolve:
            _add_scope(scope)
        elif score >= min_clarify and scope not in found:
            _add_scope(scope)

    return tuple(ordered)


def extract_requested_standards(query: str, profile: AgentProfile | None = None) -> set[str]:
    return set(extract_requested_scopes(query, profile=profile))


def has_clause_reference(query: str, profile: AgentProfile | None = None) -> bool:
    """Detect if query mentions a specific clause/article using profile patterns."""
    text = query.lower()
    patterns = profile.router.reference_patterns if profile else [r"\b\d+(?:\.\d+)+\b"]

    for p in patterns:
        if re.search(p, text):
            return True
    return False


def suggest_scope_candidates(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    _, _, _, _, scope_hints, _, _ = _router_hints(profile)
    text = (query or "").strip().lower()
    ranked: list[str] = []

    # First, use the explicit scope hints from the profile
    for standard, hints in scope_hints.items():
        if any(h in text for h in hints):
            ranked.append(standard)

    # Then, use the same pattern-based extraction as extract_requested_standards
    # to find additional candidates that might not have explicit hints but match patterns.
    pattern_based_candidates = extract_requested_standards(query, profile=profile)
    for candidate in pattern_based_candidates:
        if candidate not in ranked:  # Avoid duplicates if already added by hints
            ranked.append(candidate)

    if ranked:
        seen: set[str] = set()
        ordered = [s for s in ranked if not (s in seen or seen.add(s))]
        return tuple(ordered)

    # If no candidates found by hints or patterns, return all available scope keys from hints
    if scope_hints:
        return tuple(scope_hints.keys())
    return tuple()


def detect_scope_candidates(query: str, profile: AgentProfile | None = None) -> tuple[str, ...]:
    _, _, _, _, scope_hints, _, _ = _router_hints(profile)
    requested = list(extract_requested_scopes(query, profile=profile))
    text = (query or "").strip().lower()
    for standard, hints in scope_hints.items():
        if standard in requested:
            continue
        if any(h in text for h in hints):
            requested.append(standard)
    return tuple(requested)


def detect_conflict_objectives(query: str, profile: AgentProfile | None = None) -> bool:
    _, _, _, _, _, conflict_markers, evidence_markers = _router_hints(profile)
    text = (query or "").strip().lower()
    has_conflict = any(marker in text for marker in conflict_markers)
    has_evidence = any(marker in text for marker in evidence_markers)
    return has_conflict and has_evidence


def classify_intent(query: str, profile: AgentProfile | None = None) -> QueryIntent:
    return classify_intent_with_trace(query, profile=profile)[0]


def classify_intent_with_trace(
    query: str,
    profile: AgentProfile | None = None,
) -> tuple[QueryIntent, dict[str, Any]]:
    """Return intent plus a trace payload suitable for observability."""
    if profile is not None:
        profile_match = _classify_with_profile_rules(query, profile)
        if profile_match is not None:
            return profile_match

        default_mode = str(profile.query_modes.default_mode or "").strip()
        if default_mode:
            return QueryIntent(mode=default_mode, rationale="profile_default_mode"), {
                "version": "profile_rules_v1",
                "mode": default_mode,
                "confidence": 0.5,
                "reasons": ["default_mode"],
                "features": {},
                "blocked_modes": [],
            }

    generic_mode = "default"
    return QueryIntent(mode=generic_mode, rationale="generic_default_mode"), {
        "version": "generic",
        "mode": generic_mode,
        "confidence": 0.4,
        "reasons": ["generic_default"],
        "features": {},
        "blocked_modes": [],
    }


def build_retrieval_plan(
    intent: QueryIntent,
    query: str = "",
    profile: AgentProfile | None = None,
) -> RetrievalPlan:
    requested_standards = extract_requested_scopes(query, profile=profile)

    if profile is not None and profile.query_modes.modes:
        mode_name = str(intent.mode or "").strip()
        mode_cfg = profile.query_modes.modes.get(mode_name)
        if mode_cfg is not None:
            retrieval_key = str(mode_cfg.retrieval_profile or mode_name).strip() or mode_name
            retrieval_cfg = profile.retrieval.by_mode.get(retrieval_key)
            if retrieval_cfg is not None:
                return RetrievalPlan(
                    mode=mode_name,
                    chunk_k=int(retrieval_cfg.chunk_k),
                    chunk_fetch_k=int(retrieval_cfg.chunk_fetch_k),
                    summary_k=int(retrieval_cfg.summary_k),
                    require_literal_evidence=bool(mode_cfg.require_literal_evidence),
                    allow_inference=bool(mode_cfg.allow_inference),
                    response_contract=(
                        str(mode_cfg.response_contract).strip()
                        if mode_cfg.response_contract
                        else None
                    ),
                    requested_standards=requested_standards,
                )
            return RetrievalPlan(
                mode=mode_name,
                chunk_k=30,
                chunk_fetch_k=120,
                summary_k=5,
                require_literal_evidence=bool(mode_cfg.require_literal_evidence),
                allow_inference=bool(mode_cfg.allow_inference),
                response_contract=(
                    str(mode_cfg.response_contract).strip() if mode_cfg.response_contract else None
                ),
                requested_standards=requested_standards,
            )

    mode_cfg = profile.retrieval.by_mode.get(intent.mode) if profile is not None else None
    if mode_cfg is not None:
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=int(mode_cfg.chunk_k),
            chunk_fetch_k=int(mode_cfg.chunk_fetch_k),
            summary_k=int(mode_cfg.summary_k),
            require_literal_evidence=bool(mode_cfg.require_literal_evidence),
            allow_inference=not bool(mode_cfg.require_literal_evidence),
            requested_standards=requested_standards,
        )

    if profile is not None and profile.retrieval.by_mode:
        first_cfg = next(iter(profile.retrieval.by_mode.values()))
        return RetrievalPlan(
            mode=intent.mode,
            chunk_k=int(first_cfg.chunk_k),
            chunk_fetch_k=int(first_cfg.chunk_fetch_k),
            summary_k=int(first_cfg.summary_k),
            require_literal_evidence=bool(first_cfg.require_literal_evidence),
            allow_inference=not bool(first_cfg.require_literal_evidence),
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
