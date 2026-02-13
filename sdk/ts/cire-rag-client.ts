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
  answer: string;
  citations: string[];
  mode: string;
  scope_warnings: string | null;
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
  }): Promise<CreateDocumentResponse> {
    const form = new FormData();
    form.append("file", params.file, params.filename);
    form.append("metadata", params.metadataJson);

    return this.request<CreateDocumentResponse>("POST", "/documents", {
      body: form,
    });
  }

  async listDocuments(limit = 20): Promise<ListDocumentsResponse> {
    return this.request<ListDocumentsResponse>("GET", `/documents?limit=${encodeURIComponent(String(limit))}`);
  }

  async getDocumentStatus(documentId: string): Promise<DocumentStatusResponse> {
    return this.request<DocumentStatusResponse>("GET", `/documents/${encodeURIComponent(documentId)}/status`);
  }

  async deleteDocument(documentId: string, purgeChunks = true): Promise<DeleteDocumentResponse> {
    return this.request<DeleteDocumentResponse>(
      "DELETE",
      `/documents/${encodeURIComponent(documentId)}?purge_chunks=${encodeURIComponent(String(purgeChunks))}`,
    );
  }

  async createChatCompletion(
    payload: CreateChatCompletionRequest,
  ): Promise<CreateChatCompletionResponse> {
    return this.request<CreateChatCompletionResponse>("POST", "/chat/completions", {
      body: JSON.stringify(payload),
      contentType: "application/json",
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
    );
  }

  async getTenantQueueStatus(tenantId: string): Promise<GetTenantQueueStatusResponse> {
    return this.request<GetTenantQueueStatusResponse>(
      "GET",
      `/management/queue/status?tenant_id=${encodeURIComponent(tenantId)}`,
    );
  }

  async getManagementHealth(): Promise<GetManagementHealthResponse> {
    return this.request<GetManagementHealthResponse>("GET", "/management/health");
  }

  private async request<T>(
    method: HttpMethod,
    path: string,
    options?: {
      body?: BodyInit;
      contentType?: string;
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
