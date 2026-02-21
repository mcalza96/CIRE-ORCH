from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class OrchestratorQuestionRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    clarification_context: Optional[Dict[str, Any]] = None

class OrchestratorValidateScopeRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class OrchestratorExplainRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    top_n: int = 10
    k: int = 12
    fetch_k: int = 60
    filters: Optional[Dict[str, Any]] = None

class TenantItem(BaseModel):
    id: str
    name: str

class TenantListResponse(BaseModel):
    items: List[TenantItem]

class CollectionItem(BaseModel):
    id: str
    name: str
    collection_key: Optional[str] = None

class CollectionListResponse(BaseModel):
    items: List[CollectionItem]

class AgentProfileItem(BaseModel):
    id: str
    declared_profile_id: str
    version: str
    status: str
    description: str
    owner: str

class AgentProfileListResponse(BaseModel):
    items: List[AgentProfileItem]

class TenantProfileUpdateRequest(BaseModel):
    tenant_id: str
    profile_id: Optional[str] = None
    clear: bool = False

class DevTenantCreateRequest(BaseModel):
    name: str

class DevTenantCreateResponse(BaseModel):
    tenant_id: str
    name: str
