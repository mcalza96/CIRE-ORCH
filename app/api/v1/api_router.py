from fastapi import APIRouter

from app.api.v1.routes.knowledge import router as knowledge_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(knowledge_router, prefix="/knowledge")
