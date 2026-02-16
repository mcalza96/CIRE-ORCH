from app.agent.policies import classify_intent
query = "Analice cómo la falta de definición de la responsabilidad y autoridad para 'asegurar que el sistema de gestión es conforme' (Cláusula 5.3 en las tres normas) impacta directamente en la eficacia de la 'Consulta y participación de los trabajadores' (ISO 45001 Cláusula 5.4) y cómo este fallo de liderazgo impide demostrar la mejora continua (Cláusula 10.3) basándose en el análisis de datos de satisfacción del cliente (ISO 9001 Cláusula 9.1.2) y desempeño ambiental (ISO 14001 Cláusula 9.1.1)."
intent = classify_intent(query)
print(f"Mode: {intent.mode}")
print(f"Rationale: {intent.rationale}")
