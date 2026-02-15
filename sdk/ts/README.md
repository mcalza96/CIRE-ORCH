# CIRE-RAG TypeScript SDK (base)

Base API client for product endpoints under `/api/v1`.

## Covered endpoints

- `createDocument` -> `POST /documents`
- `listDocuments` -> `GET /documents`
- `getDocumentStatus` -> `GET /documents/{document_id}/status`
- `deleteDocument` -> `DELETE /documents/{document_id}`
- `createChatCompletion` -> `POST /chat/completions`
- `submitChatFeedback` -> `POST /chat/feedback`
- `listTenantCollections` -> `GET /management/collections`
- `getTenantQueueStatus` -> `GET /management/queue/status`
- `getManagementHealth` -> `GET /management/health`
- `validateScope` -> `POST /retrieval/validate-scope`
- `retrievalHybrid` -> `POST /retrieval/hybrid`
- `retrievalMultiQuery` -> `POST /retrieval/multi-query`
- `retrievalExplain` -> `POST /retrieval/explain`

## Usage

```ts
import { CireRagClient } from "./cire-rag-client";

const client = new CireRagClient({
  baseUrl: "http://localhost:8000",
  apiKey: process.env.CIRE_RAG_API_KEY,
});

const answer = await client.createChatCompletion({
  message: "What does ISO 9001 clause 8.5 require?",
  tenant_id: "tenant-demo",
  max_context_chunks: 8,
});

console.log(answer.context_chunks, answer.citations);
```

## Error handling

All non-2xx responses throw `CireRagApiError` with:

- `status`
- `code`
- `message`
- `details`
- `requestId`
