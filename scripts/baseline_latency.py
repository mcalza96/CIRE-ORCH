import asyncio
import json
import time
import sys
import os
import random
import argparse
from pathlib import Path
from statistics import median, quantiles

# Add project roots to path so we can import apps
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.insert(0, str(project_root / "orch"))
sys.path.append(str(project_root / "rag"))

# Ensure we are in the right directory for relative imports
os.chdir(str(project_root / "orch"))

# Placeholder for HandleQuestionUseCase and dependencies
# In a real environment, we would import the actual classes
try:
    from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
    from app.agent.http_adapters import RagEngineRetrieverAdapter, GroundedAnswerAdapter
    from app.agent.grounded_answer_service import GroundedAnswerService
    from app.agent.adapters import LiteralEvidenceValidator
    from app.infrastructure.config import settings
    from app.api.deps import UserContext
    from app.cartridges.loader import get_cartridge_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# List of standards for cross-standard queries
STANDARDS = ["ISO 9001", "ISO 14001", "ISO 45001", "ISO 27001", "ISO 50001"]

def generate_queries(count=200):
    queries = []
    # 50% simple queries
    simple_templates = [
        "¿Qué dice la cláusula {clause} de {standard}?",
        "Resume los requisitos de {standard} sobre {topic}.",
        "¿Cómo se define {concept} en {standard}?",
        "¿Cuáles son los objetivos de {standard}?"
    ]
    topics = ["auditoría interna", "control operacional", "contexto de la organización", "liderazgo", "mejora continua"]
    concepts = ["no conformidad", "acción correctiva", "riesgos y oportunidades", "partes interesadas"]
    clauses = ["4.1", "5.2", "6.1", "7.5", "8.1", "9.2", "10.3"]

    for _ in range(count // 2):
        template = random.choice(simple_templates)
        standard = random.choice(STANDARDS)
        clause = random.choice(clauses)
        topic = random.choice(topics)
        concept = random.choice(concepts)
        queries.append(template.format(standard=standard, clause=clause, topic=topic, concept=concept))

    # 50% cross-standard queries
    cross_templates = [
        "Compara los requisitos de {std1} y {std2} sobre {topic}.",
        "¿Cómo se integra {std1} con {std2} en el proceso de {topic}?",
        "Diferencias entre {std1} y {std2} respecto a {concept}.",
        "Requisitos comunes de {std1}, {std2} y {std3}."
    ]

    for _ in range(count - (count // 2)):
        template = random.choice(cross_templates)
        stds = random.sample(STANDARDS, k=3)
        topic = random.choice(topics)
        concept = random.choice(concepts)
        queries.append(template.format(std1=stds[0], std2=stds[1], std3=stds[2], topic=topic, concept=concept))

    return queries

async def run_benchmark(queries, tenant_id="default-tenant", collection_id=None):
    # Setup UseCase
    retriever = RagEngineRetrieverAdapter()
    answer_generator = GroundedAnswerAdapter(service=GroundedAnswerService())
    validator = LiteralEvidenceValidator()
    use_case = HandleQuestionUseCase(
        retriever=retriever,
        answer_generator=answer_generator,
        validator=validator,
    )
    
    loader = get_cartridge_loader()
    resolved = await loader.resolve_for_tenant_async(tenant_id=tenant_id)
    agent_profile = resolved.profile

    results = []
    print(f"Starting benchmark of {len(queries)} queries for tenant {tenant_id}...")
    
    for i, query in enumerate(queries):
        start_time = time.perf_counter()
        command = HandleQuestionCommand(
            query=query,
            tenant_id=tenant_id,
            user_id="benchmark-user",
            collection_id=collection_id,
            scope_label=f"tenant={tenant_id}",
            agent_profile=agent_profile,
            profile_resolution=resolved.resolution.model_dump()
        )
        
        error = None
        context_chunks_count = 0
        multi_query_all_failed = False
        tool_timeout = False
        
        try:
            result = await use_case.execute(command)
            context_chunks_count = len(result.answer.evidence)
            
            # Check for specific failure markers in trace if available
            trace = result.retrieval.trace or {}
            if trace.get("multi_query_all_failed"):
                multi_query_all_failed = True
        except Exception as e:
            error = str(e)
            if "timeout" in error.lower():
                tool_timeout = True
            if "MULTI_QUERY_ALL_FAILED" in error:
                multi_query_all_failed = True

        duration = (time.perf_counter() - start_time) * 1000
        
        results.append({
            "query": query,
            "duration_ms": duration,
            "context_chunks": context_chunks_count,
            "tool_timeout": tool_timeout,
            "multi_query_all_failed": multi_query_all_failed,
            "error": error
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(queries)} queries...")

    return results

def calculate_metrics(results):
    durations = [r["duration_ms"] for r in results if r["error"] is None]
    if not durations:
        return {"error": "No successful queries"}
        
    p50 = median(durations)
    p95 = quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
    
    empty_context_count = sum(1 for r in results if r["context_chunks"] == 0 and r["error"] is None)
    tool_timeout_count = sum(1 for r in results if r["tool_timeout"])
    mq_failed_count = sum(1 for r in results if r["multi_query_all_failed"])
    
    total = len(results)
    
    return {
        "count": total,
        "p50_ms": p50,
        "p95_ms": p95,
        "empty_context_rate": (empty_context_count / total) * 100,
        "tool_timeout_rate": (tool_timeout_count / total) * 100,
        "multi_query_all_failed_rate": (mq_failed_count / total) * 100,
    }

async def main():
    parser = argparse.ArgumentParser(description="CIRE Benchmark Script")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries to run")
    parser.add_argument("--output", type=str, default="baseline_report.json", help="Output JSON file")
    parser.add_argument("--tenant", type=str, default="demo-tenant", help="Tenant ID")
    parser.add_argument("--collection", type=str, default=None, help="Collection ID")
    args = parser.parse_args()

    # Attempt to use real tenant if possible, or fallback
    tenant_id = os.getenv("BENCHMARK_TENANT_ID", args.tenant)
    
    queries = generate_queries(args.queries)
    results = await run_benchmark(queries, tenant_id=tenant_id, collection_id=args.collection)
    
    metrics = calculate_metrics(results)
    
    report = {
        "timestamp": time.time(),
        "metrics": metrics,
        "results": results
    }
    
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n--- Benchmark Metrics ---")
    print(f"p50: {metrics['p50_ms']:.2f}ms")
    print(f"p95: {metrics['p95_ms']:.2f}ms")
    print(f"Context Chunks = 0: {metrics['empty_context_rate']:.2f}%")
    print(f"Tool Timeout: {metrics['tool_timeout_rate']:.2f}%")
    print(f"Multi-Query All Failed: {metrics['multi_query_all_failed_rate']:.2f}%")
    print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
