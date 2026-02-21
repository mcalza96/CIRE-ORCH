from fastapi import APIRouter

from app.api.v1.routers.knowledge import router as knowledge_router
from app.api.v1.routers.observability import router as observability_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(knowledge_router, prefix="/knowledge")
v1_router.include_router(observability_router, prefix="/observability")
