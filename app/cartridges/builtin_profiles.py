from __future__ import annotations


BUILTIN_PROFILES: dict[str, dict] = {
    "base": {
        "profile_id": "base",
        "version": "1.0.0",
        "status": "active",
        "domain_entities": ["evidencia", "fuente", "requisito"],
        "synthesis": {
            "system_persona": "Eres un analista tecnico agnostico. Responde solo con evidencia recuperada.",
            "citation_format": "C#/R#",
            "strict_subject_label": "Afirmacion",
            "strict_reference_label": "Referencia",
            "synthesis_rules": [
                "Cada afirmacion relevante debe tener soporte en el contexto recuperado.",
                "Si no existe evidencia suficiente, dilo explicitamente.",
                "No inventes fuentes ni requisitos.",
            ],
        },
        "router": {
            "literal_list_hints": ["lista", "enumera", "listado", "vinetas"],
            "literal_normative_hints": ["texto exacto", "literal", "que exige", "requisito"],
            "comparative_hints": ["compar", "difer", "vs", "entre", "respecto"],
            "interpretive_hints": ["implica", "impacto", "analiza", "causa", "por que"],
            "reference_patterns": [r"\b\d+(?:\.\d+)+\b"],
            "scope_hints": {},
            "scope_patterns": [],
        },
        "clarification_rules": [],
    },
    "iso_auditor": {
        "profile_id": "iso_auditor",
        "version": "1.0.0",
        "status": "active",
        "domain_entities": ["ISO 9001", "ISO 14001", "ISO 45001", "NC", "accion correctiva"],
        "synthesis": {
            "system_persona": "Eres un auditor trinorma. Responde con trazabilidad estricta usando evidencia del contexto recuperado.",
            "citation_format": "C#/R#",
            "strict_subject_label": "Clausula",
            "strict_reference_label": "Fuente (C#/R#)",
            "synthesis_rules": [
                "Prioriza evidencia explicita de clausulas y requisitos.",
                "Si no hay evidencia suficiente, indicalo sin inventar contenido normativo.",
                "Manten lenguaje tecnico y verificable.",
            ],
        },
        "router": {
            "scope_hints": {
                "ISO 9001": ["calidad", "cliente", "producto", "servicio"],
                "ISO 14001": ["ambient", "legal", "cumplimiento", "aspecto ambiental"],
                "ISO 45001": ["seguridad", "salud", "sst", "riesgo laboral", "trabajador"],
            },
            "scope_patterns": [
                {"label": "ISO 9001", "regex": r"\biso\s*[-:]?\s*9001\b|\b9001\b"},
                {"label": "ISO 14001", "regex": r"\biso\s*[-:]?\s*14001\b|\b14001\b"},
                {"label": "ISO 45001", "regex": r"\biso\s*[-:]?\s*45001\b|\b45001\b"},
            ],
            "reference_patterns": [r"\b\d+(?:\.\d+)+\b"],
        },
        "clarification_rules": [
            {
                "rule_id": "denunciante_vs_trazabilidad",
                "min_scope_count": 2,
                "all_markers": ["conflicto"],
                "any_markers": ["denuncia", "confidencial", "represalia", "trazabilidad"],
                "question_template": "Detecte conflicto entre objetivos en un escenario multi-scope ({scopes}). ¿Priorizo proteccion del denunciante, forense de trazabilidad, o balanceado?",
                "options": [
                    "Proteccion al denunciante",
                    "Forense de trazabilidad",
                    "Balanceado",
                ],
            },
            {
                "rule_id": "multi_scope_disambiguation",
                "min_scope_count": 2,
                "any_markers": ["analiza", "impacto", "diferencia"],
                "question_template": "Detecte senales de multiples alcances ({scopes}). ¿Quieres analisis integrado o limitarlo a un alcance especifico?",
                "options": ["Analisis integrado"]
            },
            {
                "rule_id": "structural_ambiguity",
                "all_markers": ["__mode__=ambigua_scope"],
                "question_template": "Necesito desambiguar el alcance antes de responder con trazabilidad. Indica el alcance objetivo (sugeridos: {scopes}).",
                "options": []
            }
        ],
    },
    "legal_cl": {
        "profile_id": "legal_cl",
        "version": "1.0.0",
        "status": "draft",
        "domain_entities": ["codigo civil", "codigo del trabajo", "articulo", "jurisprudencia"],
        "synthesis": {
            "system_persona": "Eres un analista legal chileno. Responde con evidencia recuperada y explicita cuando una conclusion sea inferencia.",
            "citation_format": "Art. X / Fuente",
            "strict_subject_label": "Articulo",
            "strict_reference_label": "Referencia legal",
            "synthesis_rules": [
                "Prioriza norma vigente y jerarquia normativa.",
                "Si hay conflicto normativo, explicitalo.",
                "No inventes articulos ni interpretaciones sin soporte.",
            ],
        },
        "router": {
            "scope_hints": {
                "codigo civil": ["civil", "contrato", "obligaciones"],
                "codigo del trabajo": ["laboral", "trabajador", "empleador"],
            },
            "scope_patterns": [
                {"label": "codigo civil", "regex": r"\bcodigo\s+civil\b"},
                {"label": "codigo del trabajo", "regex": r"\bcodigo\s+del\s+trabajo\b"},
            ],
            "reference_patterns": [r"\bart(?:iculo)?\s*\d+(?:\s*letra\s*[a-z])?\b"],
        },
    },
}
