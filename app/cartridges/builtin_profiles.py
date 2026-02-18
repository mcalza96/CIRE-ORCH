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
        "query_modes": {
            "default_mode": "explanatory_response",
            "modes": {
                "literal_cross_scope_check": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_cross_scope_check",
                    "tool_hints": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                    "execution_plan": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                    "coverage_requirements": {
                        "require_all_requested_scopes": True,
                        "min_clause_refs": 0,
                    },
                    "decomposition_policy": {
                        "max_subqueries": 12,
                        "clause_focus": True,
                        "scope_focus": True,
                    },
                },
                "literal_list_extract": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_list_extract",
                    "tool_hints": [
                        "semantic_retrieval",
                        "structural_extraction",
                        "citation_validator",
                    ],
                },
                "literal_clause_check": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_clause_check",
                    "tool_hints": ["semantic_retrieval", "citation_validator"],
                },
                "cross_scope_analysis": {
                    "require_literal_evidence": False,
                    "allow_inference": True,
                    "retrieval_profile": "cross_scope_analysis",
                    "tool_hints": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                },
                "scope_clarification": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "scope_clarification",
                    "tool_hints": ["citation_validator"],
                },
                "explanatory_response": {
                    "require_literal_evidence": False,
                    "allow_inference": True,
                    "retrieval_profile": "explanatory_response",
                    "tool_hints": ["semantic_retrieval", "citation_validator"],
                },
            },
            "intent_rules": [
                {
                    "id": "base_list",
                    "mode": "literal_list_extract",
                    "any_keywords": ["lista", "enumera", "listado", "vinetas"],
                },
                {
                    "id": "base_literal",
                    "mode": "literal_clause_check",
                    "any_keywords": ["clausula", "literal", "exacto", "cita", "que exige"],
                },
                {
                    "id": "base_compare",
                    "mode": "cross_scope_analysis",
                    "any_keywords": ["compar", "difer", "vs", "entre", "respecto", "impacto"],
                },
            ],
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
        "query_modes": {
            "default_mode": "explanatory_audit",
            "modes": {
                "literal_cross_scope_check": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_cross_scope_check",
                    "tool_hints": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                    "execution_plan": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                    "coverage_requirements": {
                        "require_all_requested_scopes": True,
                        "min_clause_refs": 0,
                    },
                    "decomposition_policy": {
                        "max_subqueries": 12,
                        "clause_focus": True,
                        "scope_focus": True,
                    },
                },
                "literal_list_extract": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_list_extract",
                    "tool_hints": [
                        "semantic_retrieval",
                        "structural_extraction",
                        "citation_validator",
                    ],
                },
                "literal_clause_check": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "literal_clause_check",
                    "tool_hints": ["semantic_retrieval", "citation_validator"],
                },
                "cross_standard_analysis": {
                    "require_literal_evidence": False,
                    "allow_inference": True,
                    "retrieval_profile": "cross_standard_analysis",
                    "tool_hints": [
                        "semantic_retrieval",
                        "logical_comparison",
                        "citation_validator",
                    ],
                },
                "scope_ambiguity": {
                    "require_literal_evidence": True,
                    "allow_inference": False,
                    "retrieval_profile": "scope_ambiguity",
                    "tool_hints": ["citation_validator"],
                },
                "explanatory_audit": {
                    "require_literal_evidence": False,
                    "allow_inference": True,
                    "retrieval_profile": "explanatory_audit",
                    "tool_hints": ["semantic_retrieval", "citation_validator"],
                },
            },
            "intent_rules": [
                {
                    "id": "iso_ambiguous",
                    "mode": "scope_ambiguity",
                    "any_markers": ["__mode__=scope_ambiguity"],
                },
                {
                    "id": "iso_literal_triscope",
                    "mode": "literal_cross_scope_check",
                    "any_keywords": [
                        "textualmente",
                        "literal",
                        "c#/r#",
                        "cita",
                        "citas",
                        "que exige",
                    ],
                    "all_patterns": [
                        "\\biso\\s*[-:]?\\s*9001\\b",
                        "\\biso\\s*[-:]?\\s*14001\\b",
                        "\\biso\\s*[-:]?\\s*45001\\b",
                    ],
                },
                {
                    "id": "iso_list",
                    "mode": "literal_list_extract",
                    "any_keywords": [
                        "lista",
                        "enumera",
                        "listado",
                        "vinetas",
                        "entradas",
                        "salidas",
                    ],
                },
                {
                    "id": "iso_literal",
                    "mode": "literal_clause_check",
                    "any_keywords": [
                        "clausula",
                        "literal",
                        "textualmente",
                        "verbatim",
                        "cita",
                        "que exige",
                    ],
                },
                {
                    "id": "iso_compare",
                    "mode": "cross_standard_analysis",
                    "any_keywords": [
                        "compar",
                        "difer",
                        "vs",
                        "ambas",
                        "respecto",
                        "impacto",
                        "interaccion",
                    ],
                },
            ],
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
                "options": ["Analisis integrado"],
            },
            {
                "rule_id": "structural_ambiguity",
                "all_markers": ["__mode__=scope_ambiguity"],
                "question_template": "Necesito desambiguar el alcance antes de responder con trazabilidad. Indica el alcance objetivo (sugeridos: {scopes}).",
                "options": [],
            },
        ],
        "retrieval": {
            "min_score": 0.72,
            "search_hints": [],
            "by_mode": {
                "literal_cross_scope_check": {
                    "chunk_k": 60,
                    "chunk_fetch_k": 340,
                    "summary_k": 3,
                    "require_literal_evidence": True,
                },
                "literal_list_extract": {
                    "chunk_k": 45,
                    "chunk_fetch_k": 220,
                    "summary_k": 3,
                    "require_literal_evidence": True,
                },
                "literal_clause_check": {
                    "chunk_k": 55,
                    "chunk_fetch_k": 300,
                    "summary_k": 3,
                    "require_literal_evidence": True,
                },
                "cross_standard_analysis": {
                    "chunk_k": 35,
                    "chunk_fetch_k": 140,
                    "summary_k": 5,
                    "require_literal_evidence": False,
                },
                "scope_ambiguity": {
                    "chunk_k": 0,
                    "chunk_fetch_k": 0,
                    "summary_k": 0,
                    "require_literal_evidence": True,
                },
                "explanatory_audit": {
                    "chunk_k": 30,
                    "chunk_fetch_k": 120,
                    "summary_k": 5,
                    "require_literal_evidence": False,
                },
            },
        },
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
