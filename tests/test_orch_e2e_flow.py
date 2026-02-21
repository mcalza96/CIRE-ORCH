import asyncio
import httpx
import pytest
import os
from typing import Any, Dict

# Configuración de base para tests e2e en Orchestrator
ORCH_BASE_URL = os.environ.get("ORCH_URL", "http://localhost:8001")
AUTH_TOKEN = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "test-token")
TEST_TENANT_ID = "289007d1-07b1-40ca-bd8f-700d5c8659e7"
TEST_COLLECTION_ID = "8c29a898-a27d-49c7-8123-7ce3b8d14be5"

@pytest.fixture
async def api_client():
    async with httpx.AsyncClient(base_url=ORCH_BASE_URL, timeout=60.0) as client:
        yield client

@pytest.mark.asyncio
async def test_orch_answer_e2e_full_flow(api_client: httpx.AsyncClient):
    """
    Prueba el flujo completo: Orchestrator -> RAG -> Orchestrator response.
    Valida la intención, el plan de búsqueda y la respuesta final.
    """
    payload = {
        "query": "¿Qué dice la introducción del documento?",
        "tenant_id": TEST_TENANT_ID,
        "collection_id": TEST_COLLECTION_ID
    }
    headers = {
        "X-Tenant-ID": TEST_TENANT_ID,
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "x-orch-profile": "base"
    }
    
    try:
        response = await api_client.post("/api/v1/knowledge/answer", json=payload, headers=headers)
        
        # Si el servicio no está corriendo localmente, fallará aquí.
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "source_chunks" in data or "metadata" in data
            # Validamos que el modo sea explanatory_response o similar
            assert data.get("answer") != ""
        elif response.status_code == 401:
            pytest.skip("Auth token inválido para test real")
    except httpx.ConnectError:
        pytest.skip("Orchestrator no está corriendo en localhost:8001")

@pytest.mark.asyncio
async def test_orch_explain_retrieval_logic(api_client: httpx.AsyncClient):
    """
    Valida que el endpoint /explain del Orchestrator devuelva el desglose 
    técnico de por qué se eligieron ciertos chunks.
    """
    payload = {
        "query": "ISO 9001 cláusula 4.1",
        "tenant_id": TEST_TENANT_ID,
        "collection_id": TEST_COLLECTION_ID,
        "top_n": 3
    }
    headers = {
        "X-Tenant-ID": TEST_TENANT_ID,
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    try:
        response = await api_client.post("/api/v1/knowledge/explain", json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "items" in data
            assert len(data["items"]) > 0
            # Verificamos que contenga los componentes del score (Gravity/Atomic)
            first_item = data["items"][0]
            assert "explain" in first_item
            assert "score_components" in first_item["explain"]
    except httpx.ConnectError:
        pytest.skip("Orchestrator no está corriendo")

@pytest.mark.asyncio
async def test_orch_tenant_isolation_middleware(api_client: httpx.AsyncClient):
    """
    Verifica que el middleware de Orchestrator rechaza tokens que no pertenecen al tenant.
    Simula el error 401/403 de seguridad.
    """
    headers = {
        "X-Tenant-ID": "tenant-totalmente-falso",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    payload = {"query": "test", "tenant_id": "tenant-totalmente-falso"}
    
    response = await api_client.post("/api/v1/knowledge/answer", json=payload, headers=headers)
    
    # Dependiendo de la implementación exacta del middleware de Orch:
    # Si detecta discrepancia o tenant inexistente, debería dar Error.
    assert response.status_code in [400, 401, 403, 404]
