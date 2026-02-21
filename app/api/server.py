import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse

from app.api.v1.api_router import v1_router
from app.profiles.loader import get_profile_loader
from app.infrastructure.config import settings
from app.infrastructure.clients.rag_client import build_rag_http_client

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO), format="%(message)s"
)
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_profile_loader().validate_profile_files_strict()
    app.state.rag_http_client = build_rag_http_client()
    try:
        yield
    finally:
        rag_http_client = getattr(app.state, "rag_http_client", None)
        if rag_http_client is not None:
            await rag_http_client.aclose()


app = FastAPI(
    title="Q/A Orchestrator API",
    description="Q/A orchestration API backed by external rag-engine retrieval contracts.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    logger.error(
        "orchestrator_backend_contract_breach",
        type="contract_violation",
        direction="outbound_backend",
        endpoint=str(request.url),
        validation_errors=exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal Server Error: Data Contract Breach",
            "type": "backend_contract_breach",
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "orchestrator_frontend_contract_breach",
        type="contract_violation",
        direction="inbound_backend",
        endpoint=str(request.url),
        validation_errors=exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "type": "frontend_contract_breach"},
    )


app.include_router(v1_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "qa-orchestrator", "api_v1": "available"}
