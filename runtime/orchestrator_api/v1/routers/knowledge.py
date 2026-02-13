from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from runtime.config import settings
from runtime.qa_orchestrator.adapters import LiteralEvidenceValidator
from runtime.qa_orchestrator.application import HandleQuestionCommand, HandleQuestionUseCase
from runtime.qa_orchestrator.grounded_answer_service import GroundedAnswerService
from runtime.qa_orchestrator.http_adapters import GroundedAnswerAdapter, RagEngineRetrieverAdapter
from runtime.scope_metrics import scope_metrics_store

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["knowledge"])


class OrchestratorQuestionRequest(BaseModel):
    query: str
    tenant_id: str
    collection_id: Optional[str] = None


def _build_use_case() -> HandleQuestionUseCase:
    rag_base_url = str(settings.RAG_ENGINE_URL or "http://localhost:8000").strip()
    retriever = RagEngineRetrieverAdapter(base_url=rag_base_url)
    answer_generator = GroundedAnswerAdapter(service=GroundedAnswerService())
    validator = LiteralEvidenceValidator()
    return HandleQuestionUseCase(
        retriever=retriever,
        answer_generator=answer_generator,
        validator=validator,
    )


@router.post("/answer", response_model=Dict[str, Any])
async def answer_with_orchestrator(
    request: OrchestratorQuestionRequest,
    use_case: HandleQuestionUseCase = Depends(_build_use_case),
):
    try:
        scope_metrics_store.record_request(request.tenant_id)
        if not request.tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required")

        result = await use_case.execute(
            HandleQuestionCommand(
                query=request.query,
                tenant_id=request.tenant_id,
                collection_id=request.collection_id,
                scope_label=f"tenant={request.tenant_id}",
            )
        )

        if result.clarification:
            scope_metrics_store.record_clarification(request.tenant_id)

        blocked = (
            not result.validation.accepted
            and any("Scope mismatch" in issue for issue in result.validation.issues)
        )
        if blocked:
            scope_metrics_store.record_mismatch_detected(request.tenant_id)
            scope_metrics_store.record_mismatch_blocked(request.tenant_id)

        return {
            "answer": result.answer.text,
            "mode": result.plan.mode,
            "citations": [item.source for item in result.answer.evidence],
            "context_chunks": [item.content for item in result.answer.evidence],
            "requested_scopes": list(result.plan.requested_standards),
            "clarification": (
                {
                    "question": result.clarification.question,
                    "options": list(result.clarification.options),
                }
                if result.clarification
                else None
            ),
            "validation": {
                "accepted": result.validation.accepted,
                "issues": list(result.validation.issues),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("orchestrator_answer_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Orchestrator answer failed")


@router.get("/scope-health", response_model=Dict[str, Any])
async def scope_health(tenant_id: Optional[str] = Query(default=None)):
    return scope_metrics_store.snapshot(tenant_id=tenant_id)
