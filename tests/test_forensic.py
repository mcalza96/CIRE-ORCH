import asyncio
import httpx
import json
import os

token = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
tenant_id = "5cdcd14c-c256-41b3-ade0-f93b73d71429" # Assuming test tenant or generic

async def main():
    query = """Para cumplir un objetivo estratégico de "Cero Emisiones" (ISO 14001, Cláusula 6.2) exigido como requisito excluyente por su cliente más importante (ISO 9001, Cláusula 8.2.2), la Alta Dirección ordena sustituir inmediatamente la materia prima por un compuesto ecológico. Sin embargo, los operadores paralizan la producción amparándose en su derecho a retirarse de un peligro inminente (ISO 45001, Cláusula 5.4), ya que el nuevo compuesto ecológico genera vapores asfixiantes al entrar en las máquinas. Analice: ¿Cómo este escenario demuestra que la Alta Dirección utilizó el pretexto de la "Mejora Continua" ambiental (ISO 14001, 10.3) y el "Enfoque al Cliente" (ISO 9001, 5.1.2) para evadir la "Gestión del Cambio" (ISO 45001, 8.1.3)? En consecuencia, ¿por qué esta decisión convierte la liberación del producto en un incumplimiento del "Control de Producción" (ISO 9001, 8.5.1)?"""
    
    url = "http://localhost:8001/api/v1/knowledge/answer"
    headers = {
        "X-Tenant-ID": tenant_id,
        "Authorization": f"Bearer {token}",
        "x-orch-profile": "base",
        "x-orch-mode": "cross_scope_analysis"
    }
    payload = {
        "query": query,
        "tenant_id": tenant_id,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, json=payload, headers=headers)
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            print("--- ANSWER ---")
            print(data.get("answer"))
            print("--- Retrieval TRACE ---")
            print(data.get("retrieval_plan", {}).get("subqueries"))
            print(data.get("retrieval_plan", {}).get("kernel_flags"))
        else:
            print(res.text)

if __name__ == "__main__":
    asyncio.run(main())
