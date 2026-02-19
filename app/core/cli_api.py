from __future__ import annotations

import json
import httpx
import logging
from typing import Any, Callable

from app.core.config import settings
from sdk.python.cire_rag_sdk import (
    TENANT_MISMATCH_CODE,
    TenantContext,
    TenantProtocolError,
    TenantSelectionRequiredError,
    user_message_for_tenant_error_code,
)

logger = logging.getLogger(__name__)


def parse_error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        if isinstance(data.get("error"), dict):
            return data["error"]
        if isinstance(data.get("detail"), dict):
            return data["detail"]
    return {}


async def post_answer(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_context: TenantContext,
    query: str,
    collection_id: str | None,
    agent_profile_id: str | None,
    access_token: str | None = None,
    retry_on_mismatch: bool = True,
) -> dict[str, Any]:
    resolved_tenant = tenant_context.get_tenant()
    if not resolved_tenant:
        raise TenantSelectionRequiredError()

    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": resolved_tenant,
    }
    if collection_id:
        payload["collection_id"] = collection_id

    headers = {"X-Tenant-ID": resolved_tenant}
    profile_header = str(settings.ORCH_AGENT_PROFILE_HEADER or "X-Agent-Profile").strip()
    if profile_header and str(agent_profile_id or "").strip():
        headers[profile_header] = str(agent_profile_id).strip()
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = await client.post(
        orchestrator_url.rstrip("/") + "/api/v1/knowledge/answer",
        json=payload,
        headers=headers,
    )
    if response.status_code >= 400:
        error = parse_error_payload(response)
        code = str(error.get("code") or "")
        request_id = str(
            error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown"
        )
        if code == TENANT_MISMATCH_CODE and retry_on_mismatch and tenant_context.storage_path:
            logger.warning(f"tenant_mismatch_detected request_id={request_id}")
            previous = resolved_tenant
            reloaded = tenant_context.reload()
            if reloaded and reloaded != previous:
                return await post_answer(
                    client=client,
                    orchestrator_url=orchestrator_url,
                    tenant_context=tenant_context,
                    query=query,
                    collection_id=collection_id,
                    agent_profile_id=agent_profile_id,
                    access_token=access_token,
                    retry_on_mismatch=False,
                )
        if code:
            raise TenantProtocolError(
                status=response.status_code,
                code=code,
                message=str(error.get("message") or response.text),
                user_message=user_message_for_tenant_error_code(code),
                request_id=request_id,
                details=error.get("details"),
            )
        response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


async def post_explain(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_context: TenantContext,
    query: str,
    collection_id: str | None,
    agent_profile_id: str | None,
    access_token: str | None = None,
) -> dict[str, Any]:
    resolved_tenant = tenant_context.get_tenant()
    if not resolved_tenant:
        raise TenantSelectionRequiredError()

    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": resolved_tenant,
        "collection_id": collection_id,
        "top_n": 10,
        "k": 12,
        "fetch_k": 60,
        "filters": None,
    }
    headers = {"X-Tenant-ID": resolved_tenant}
    profile_header = str(settings.ORCH_AGENT_PROFILE_HEADER or "X-Agent-Profile").strip()
    if profile_header and str(agent_profile_id or "").strip():
        headers[profile_header] = str(agent_profile_id).strip()
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = await client.post(
        orchestrator_url.rstrip("/") + "/api/v1/knowledge/explain-retrieval",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


async def post_answer_stream(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_context: TenantContext,
    query: str,
    collection_id: str | None,
    agent_profile_id: str | None,
    access_token: str | None = None,
    on_status: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    resolved_tenant = tenant_context.get_tenant()
    if not resolved_tenant:
        raise TenantSelectionRequiredError()

    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": resolved_tenant,
    }
    if collection_id:
        payload["collection_id"] = collection_id

    headers = {"X-Tenant-ID": resolved_tenant}
    profile_header = str(settings.ORCH_AGENT_PROFILE_HEADER or "X-Agent-Profile").strip()
    if profile_header and str(agent_profile_id or "").strip():
        headers[profile_header] = str(agent_profile_id).strip()
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = orchestrator_url.rstrip("/") + "/api/v1/knowledge/answer/stream"
    final_payload: dict[str, Any] = {}
    current_event = ""

    async with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code >= 400:
            error = parse_error_payload(response)
            code = str(error.get("code") or "")
            request_id = str(
                error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown"
            )
            if code:
                raise TenantProtocolError(
                    status=response.status_code,
                    code=code,
                    message=str(error.get("message") or response.text),
                    user_message=user_message_for_tenant_error_code(code),
                    request_id=request_id,
                    details=error.get("details"),
                )
            response.raise_for_status()

        async for raw_line in response.aiter_lines():
            line = str(raw_line or "")
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                raw_data = line.split(":", 1)[1].strip()
                try:
                    payload_data = json.loads(raw_data) if raw_data else {}
                except Exception:
                    payload_data = {"raw": raw_data}

                if current_event == "status" and callable(on_status):
                    if isinstance(payload_data, dict):
                        on_status(payload_data)
                elif current_event == "result" and isinstance(payload_data, dict):
                    final_payload = payload_data
                elif current_event == "error":
                    code = (
                        str(payload_data.get("code") or "")
                        if isinstance(payload_data, dict)
                        else "ORCH_STREAM_ERROR"
                    )
                    message = (
                        str(payload_data.get("message") or "Orchestrator stream failed")
                        if isinstance(payload_data, dict)
                        else "Orchestrator stream failed"
                    )
                    request_id = (
                        str(payload_data.get("request_id") or "unknown")
                        if isinstance(payload_data, dict)
                        else "unknown"
                    )
                    raise TenantProtocolError(
                        status=500,
                        code=code,
                        message=message,
                        user_message=user_message_for_tenant_error_code(code),
                        request_id=request_id,
                    )
                continue

    return final_payload if isinstance(final_payload, dict) else {}
