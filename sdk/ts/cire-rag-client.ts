export type HttpMethod = "GET" | "POST" | "DELETE";

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details: unknown;
    request_id: string;
  };
}

export class CireRagApiError extends Error {
  public readonly status: number;
  public readonly code: string;
  public readonly details: unknown;
  public readonly requestId: string;

  constructor(status: number, payload: ApiErrorEnvelope) {
    super(payload.error.message);
    this.name = "CireRagApiError";
    this.status = status;
    this.code = payload.error.code;
    this.details = payload.error.details;
    this.requestId = payload.error.request_id;
  }
}

export interface QueueSnapshot {
  queue_depth: number;
  max_pending: number | null;
  estimated_wait_seconds: number;
}

export interface CreateDocumentResponse {
  status: "accepted";
  message: string;
  document_id: string;
  queue: QueueSnapshot;
}

export interface ListDocumentsResponse {
  items: Array<Record<string, unknown>>;
}

export interface DocumentStatusResponse {
  document_id: string;
  status: string;
  error_message: string | null;
  updated_at: string | null;
}

export interface DeleteDocumentResponse {
  status: "deleted";
  document_id: string;
  purge_chunks: boolean;
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface CreateChatCompletionRequest {
  message: string;
  tenant_id: string;
  history?: ChatMessage[];
  max_context_chunks?: number;
}

export interface CreateChatCompletionResponse {
  interaction_id: string;
  query?: string;
  context_chunks?: string[];
  context_map?: Record<string, unknown>;
  citations: string[];
  mode: string;
  requires_scope_clarification?: boolean;
  scope_warnings: string | null;
}

export type TimeRangeField = "created_at" | "updated_at";

export interface TimeRangeFilter {
  field: TimeRangeField;
  from?: string | null;
  to?: string | null;
}

export interface ScopeFilters {
  metadata?: Record<string, unknown> | null;
  time_range?: TimeRangeFilter | null;
  source_standard?: string | null;
  source_standards?: string[] | null;
}

export interface RerankOptions {
  enabled?: boolean;
}

export interface GraphOptions {
  relation_types?: string[] | null;
  node_types?: string[] | null;
  max_hops?: number | null;
}

export interface RetrievalItem {
  source: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface HybridTrace {
  filters_applied: Record<string, unknown>;
  engine_mode: string;
  planner_used: boolean;
  planner_multihop: boolean;
  fallback_used: boolean;
  timings_ms: Record<string, number>;
  warnings: string[];
}

export interface HybridRetrievalRequest {
  query: string;
  tenant_id: string;
  collection_id?: string | null;
  k?: number;
  fetch_k?: number;
  filters?: ScopeFilters | null;
  rerank?: RerankOptions | null;
  graph?: GraphOptions | null;
}

export interface HybridRetrievalResponse {
  items: RetrievalItem[];
  trace: HybridTrace;
}

export interface SubQueryRequest {
  id: string;
  query: string;
  k?: number | null;
  fetch_k?: number | null;
  filters?: ScopeFilters | null;
}

export interface MergeOptions {
  strategy?: "rrf";
  rrf_k?: number;
  top_k?: number;
}

export interface MultiQueryRetrievalRequest {
  tenant_id: string;
  collection_id?: string | null;
  queries: SubQueryRequest[];
  merge?: MergeOptions;
}

export interface SubQueryExecution {
  id: string;
  status: "ok" | "error";
  items_count: number;
  latency_ms: number;
  error_code?: string | null;
  error_message?: string | null;
}

export interface MultiQueryTrace {
  merge_strategy: string;
  rrf_k: number;
  failed_count: number;
  timings_ms: Record<string, number>;
}

export interface MultiQueryRetrievalResponse {
  items: RetrievalItem[];
  subqueries: SubQueryExecution[];
  partial: boolean;
  trace: MultiQueryTrace;
}

export interface ExplainRetrievalRequest extends HybridRetrievalRequest {
  top_n?: number;
}

export interface ScoreComponents {
  base_similarity: number;
  jina_relevance_score?: number | null;
  final_score: number;
  scope_penalized: boolean;
  scope_penalty_ratio?: number | null;
}

export interface RetrievalPath {
  source_layer?: string | null;
  source_type?: string | null;
}

export interface MatchedFilters {
  collection_id_match?: boolean | null;
  time_range_match?: boolean | null;
  metadata_keys_matched: string[];
}

export interface ExplainedItemDetails {
  score_components: ScoreComponents;
  retrieval_path: RetrievalPath;
  matched_filters: MatchedFilters;
}

export interface ExplainedRetrievalItem extends RetrievalItem {
  explain: ExplainedItemDetails;
}

export interface ExplainTrace extends HybridTrace {
  top_n: number;
}

export interface ExplainRetrievalResponse {
  items: ExplainedRetrievalItem[];
  trace: ExplainTrace;
}

export interface ScopeIssue {
  code: string;
  field: string;
  message: string;
}

export interface QueryScopeSummary {
  requested_standards: string[];
  requires_scope_clarification: boolean;
  suggested_scopes: string[];
}

export interface ValidateScopeRequest {
  query: string;
  tenant_id: string;
  collection_id?: string | null;
  filters?: ScopeFilters | null;
}

export interface ValidateScopeResponse {
  valid: boolean;
  normalized_scope: Record<string, unknown>;
  violations: ScopeIssue[];
  warnings: ScopeIssue[];
  query_scope: QueryScopeSummary;
}

export interface SubmitChatFeedbackRequest {
  interaction_id: string;
  rating: string;
  comment?: string;
}

export interface SubmitChatFeedbackResponse {
  status: "accepted";
  interaction_id: string;
}

export interface ListTenantCollectionsResponse {
  items: Array<Record<string, unknown>>;
}

export interface GetTenantQueueStatusResponse {
  status: "ok";
  tenant_id: string;
  queue: QueueSnapshot;
}

export interface GetManagementHealthResponse {
  status: "ok";
  service: string;
  api_v1: "available";
}

export interface CireRagClientConfig {
  baseUrl: string;
  apiKey?: string;
  defaultHeaders?: Record<string, string>;
  fetchImpl?: typeof fetch;
}

export class CireRagClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly defaultHeaders: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor(config: CireRagClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.defaultHeaders = config.defaultHeaders ?? {};
    this.fetchImpl = config.fetchImpl ?? fetch;
  }

  async createDocument(params: {
    file: Blob;
    filename: string;
    metadataJson: string;
    tenantId?: string;
  }): Promise<CreateDocumentResponse> {
    const form = new FormData();
    form.append("file", params.file, params.filename);
    form.append("metadata", params.metadataJson);

    return this.request<CreateDocumentResponse>("POST", "/documents", {
      body: form,
      tenantId: params.tenantId,
    });
  }

  async listDocuments(limit = 20, tenantId?: string): Promise<ListDocumentsResponse> {
    return this.request<ListDocumentsResponse>("GET", `/documents?limit=${encodeURIComponent(String(limit))}`, { tenantId });
  }

  async getDocumentStatus(documentId: string, tenantId?: string): Promise<DocumentStatusResponse> {
    return this.request<DocumentStatusResponse>("GET", `/documents/${encodeURIComponent(documentId)}/status`, { tenantId });
  }

  async deleteDocument(documentId: string, purgeChunks = true, tenantId?: string): Promise<DeleteDocumentResponse> {
    return this.request<DeleteDocumentResponse>(
      "DELETE",
      `/documents/${encodeURIComponent(documentId)}?purge_chunks=${encodeURIComponent(String(purgeChunks))}`,
      { tenantId },
    );
  }

  async createChatCompletion(
    payload: CreateChatCompletionRequest,
  ): Promise<CreateChatCompletionResponse> {
    return this.request<CreateChatCompletionResponse>("POST", "/chat/completions", {
      body: JSON.stringify(payload),
      contentType: "application/json",
      tenantId: payload.tenant_id,
    });
  }

  async submitChatFeedback(payload: SubmitChatFeedbackRequest): Promise<SubmitChatFeedbackResponse> {
    return this.request<SubmitChatFeedbackResponse>("POST", "/chat/feedback", {
      body: JSON.stringify(payload),
      contentType: "application/json",
    });
  }

  async listTenantCollections(tenantId: string): Promise<ListTenantCollectionsResponse> {
    return this.request<ListTenantCollectionsResponse>(
      "GET",
      `/management/collections?tenant_id=${encodeURIComponent(tenantId)}`,
      { tenantId },
    );
  }

  async getTenantQueueStatus(tenantId: string): Promise<GetTenantQueueStatusResponse> {
    return this.request<GetTenantQueueStatusResponse>(
      "GET",
      `/management/queue/status?tenant_id=${encodeURIComponent(tenantId)}`,
      { tenantId },
    );
  }

  async getManagementHealth(tenantId?: string): Promise<GetManagementHealthResponse> {
    return this.request<GetManagementHealthResponse>("GET", "/management/health", { tenantId });
  }

  async validateScope(payload: ValidateScopeRequest): Promise<ValidateScopeResponse> {
    return this.request<ValidateScopeResponse>("POST", "/retrieval/validate-scope", {
      body: JSON.stringify(payload),
      contentType: "application/json",
      tenantId: payload.tenant_id,
    });
  }

  async retrievalHybrid(payload: HybridRetrievalRequest): Promise<HybridRetrievalResponse> {
    return this.request<HybridRetrievalResponse>("POST", "/retrieval/hybrid", {
      body: JSON.stringify(payload),
      contentType: "application/json",
      tenantId: payload.tenant_id,
    });
  }

  async retrievalMultiQuery(payload: MultiQueryRetrievalRequest): Promise<MultiQueryRetrievalResponse> {
    return this.request<MultiQueryRetrievalResponse>("POST", "/retrieval/multi-query", {
      body: JSON.stringify(payload),
      contentType: "application/json",
      tenantId: payload.tenant_id,
    });
  }

  async retrievalExplain(payload: ExplainRetrievalRequest): Promise<ExplainRetrievalResponse> {
    return this.request<ExplainRetrievalResponse>("POST", "/retrieval/explain", {
      body: JSON.stringify(payload),
      contentType: "application/json",
      tenantId: payload.tenant_id,
    });
  }

  private async request<T>(
    method: HttpMethod,
    path: string,
    options?: {
      body?: BodyInit;
      contentType?: string;
      tenantId?: string;
    },
  ): Promise<T> {
    const headers: Record<string, string> = {
      ...this.defaultHeaders,
    };

    if (this.apiKey && !headers.Authorization) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    if (options?.contentType) {
      headers["Content-Type"] = options.contentType;
    }
    if (options?.tenantId && !headers["X-Tenant-ID"]) {
      headers["X-Tenant-ID"] = options.tenantId;
    }

    const response = await this.fetchImpl(`${this.baseUrl}/api/v1${path}`, {
      method,
      headers,
      body: options?.body,
    });

    if (!response.ok) {
      const payload = (await response.json()) as ApiErrorEnvelope;
      throw new CireRagApiError(response.status, payload);
    }

    return (await response.json()) as T;
  }
}
