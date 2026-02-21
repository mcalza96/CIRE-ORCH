Módulo 1: Enrutamiento y Definición de Perfiles (Cartuchos)
Antes de procesar la consulta, el sistema configura el "chasis" de razonamiento. La calidad de la respuesta dependerá de qué tan bien estén definidas las reglas de negocio del perfil seleccionado.
Flujo: Petición HTTP → Autenticación/Tenant → Carga de Cartucho (ISO, Legal, Analista de Laboratorio) → Inyección de Prompts y Reglas de Comportamiento.
Archivos clave a revisar:
app/cartridges/loader.py y app/cartridges/models.py (Lógica de carga y parseo).
app/cartridges/iso_auditor.yaml, app/cartridges/legal_cl.yaml (Definición estática de las reglas).
app/security/tenant_authorizer.py (Aislamiento de la data por cliente).
¿Qué analizar para mejorar?
Caché de perfiles: Si la carga de los YAML desde disco o base de datos ocurre en cada petición, estarás sumando milisegundos valiosos. Revisa el loader.py para asegurar que los perfiles activos residan en memoria.
Densidad de los Prompts: Si los archivos YAML son demasiado extensos o ambiguos, el LLM consumirá más tokens de contexto y se volverá más lento. Optimiza la redacción de las directrices de cada rol.
Módulo 2: Planificación y Comprensión de la Consulta (Query Understanding & Planning)
El sistema analiza la consulta del usuario, el historial, y decide cómo atacarla dividiéndola en sub-tareas o formulando estrategias de búsqueda para enviar al motor RAG.
Flujo: Historial + Nueva Consulta → Descomposición de la consulta → Generación del plan de recuperación (Retrieval Planner) → Activación de Nodos del Grafo.
Archivos clave a revisar:
app/graph/universal/nodes/planning.py (Punto de entrada de la lógica del grafo para este paso).
app/agent/components/query_decomposer.py (División de consultas complejas).
app/agent/retrieval_planner.py y app/agent/policies/query_analysis.py.
¿Qué analizar para mejorar?
Desambiguación y Coreferencia: Si el usuario pregunta "y qué dice sobre las multas", asegúrate de que el planificador traduzca esto a "multas asociadas al artículo X mencionado en el turno anterior".
Sobre-planificación: Revisa si el LLM está creando planes de 5 pasos para preguntas que podrían responderse en 1. Ajustar el prompt de planning.py puede ahorrar iteraciones innecesarias del grafo.
Módulo 3: Ejecución de Recuperación y Uso de Herramientas (Retrieval Execution)
Aquí es donde el orquestador interactúa con el mundo exterior (motores RAG, APIs) utilizando herramientas concretas y evalúa si lo que obtuvo es suficiente.
Flujo: Nodo de Ejecución → Selección de Herramienta → Llamada al cliente RAG externo / Herramienta Lógica → Evaluación de Suficiencia → Reintento (si falla).
Archivos clave a revisar:
app/graph/universal/nodes/execution.py y app/agent/tools/semantic_retrieval.py.
app/core/rag_retrieval_contract_client.py (El puente que se comunica con tu base de datos documental externa).
app/agent/retrieval_sufficiency_evaluator.py y app/agent/policies/retry_policy.py.
¿Qué analizar para mejorar?
Criterios de Parada (Sufficiency): El retrieval_sufficiency_evaluator.py es crítico. Si es muy estricto, el orquestador entrará en bucles de reintento buscando información inexistente. Si es muy laxo, responderá con información incompleta.
Timeouts y Concurrencia: Optimiza el cliente HTTP en rag_retrieval_contract_client.py. Si el motor RAG demora, el orquestador se bloquea. Implementar llamadas asíncronas paralelas cuando el planificador arroja múltiples sub-consultas reducirá drásticamente la latencia.
Módulo 4: Síntesis, Generación y Reflexión (Generation & Guardrails)
El motor toma la información cruda recuperada, redacta la respuesta final basándose en el perfil, y verifica que cumpla las reglas impuestas antes de entregarla al usuario.
Flujo: Resultados de Ejecución → Nodo de Generación → Validación de Citas / Reflexión → Respuesta (Streaming).
Archivos clave a revisar:
app/graph/universal/nodes/generation.py y app/agent/components/synthesis.py.
app/graph/universal/nodes/reflection.py y app/agent/tools/citation_validator.py.
app/api/v1/routes/knowledge.py (Manejo de la respuesta HTTP final).
¿Qué analizar para mejorar?
Time-To-First-Token (TTFT): La generación debe comenzar a enviar "chunks" al usuario lo antes posible. Revisa que el nodo de generación esté bien acoplado a un generador asíncrono en FastAPI.
Fricción en la Reflexión: El nodo de reflexión es un "cuello de botella" por diseño. Si el citation_validator.py detecta un error, obliga a regenerar. Optimizar el modelo base (quizás usar uno más rápido para la tarea específica de validar) acelerará el paso de control de calidad.
Módulo 5: Observabilidad y Benchmarks (Diagnóstico Continuo)
No puedes optimizar lo que no puedes medir. Este módulo no interactúa con el usuario, pero captura la telemetría de todos los módulos anteriores para identificar latencias.
Flujo: Ejecución de Grafo → Recolección de Trazas/Métricas → Ingesta asíncrona → Generación de Reportes.
Archivos clave a revisar:
app/graph/universal/trace.py y app/core/retrieval_metrics.py.
tests/evaluation/run_iso_auditor_benchmark.py y scripts/baseline_latency.py.
¿Qué analizar para mejorar?
Impacto en Rendimiento: Asegúrate de que el guardado de trazas en trace.py sea verdaderamente "fire-and-forget" (background tasks) y no bloquee el hilo principal de respuesta de FastAPI.
Frecuencia de Evaluación: Usa los scripts de tests/evaluation/ para crear una línea base de latencia (baseline). Cada vez que toques el Módulo 2 o 3, corre el benchmark para confirmar que ganaste velocidad sin perder precisión (Accuracy vs. Latency).