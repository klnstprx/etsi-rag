# Architecture
<!-- packages: all -->
<!-- related: docs/configuration.md, docs/crawler.md -->

RAG chatbot for ETSI Informatica (Universidad de Malaga) with OpenAI-compatible API.

## System Overview

```
llama-server (configurable)    <- LLM backend (OpenAI-compatible)
RustFS       (localhost:9000)  <- S3-compatible object storage
        |
        v HTTP (openai-go SDK, aws-sdk-go-v2)
+----------------------------------------------------------+
|  cmd/server/main.go  (entry point: loads config, calls   |
|                       server.New + server.Run)           |
|        |                                                 |
|        v                                                 |
|  internal/server/    (wiring, lifecycle, hot-rebuild)    |
|        |                                                 |
|        +-->  api/         (OpenAI-compatible + RAG API)  |
|        +-->  admin/       (templ + Tailwind + HTMX UI)   |
|        +-->  agent/       (tool-calling LLM loop)        |
|        +-->  rag/         (retrieval service)            |
|        +-->  crawler/     (BFS web crawler, 3-stage)     |
|        +-->  llm/         (openai-go SDK wrapper)        |
|        +-->  store/       (SQLite + cosine similarity)   |
|        +-->  objectstore/ (S3/RustFS client)             |
|        +-->  session/     (conversation state + TTL)     |
|        +-->  chunking/    (token-aware text splitting)   |
|        +-->  tokencount/  (token counting abstraction)   |
|        +-->  prompts/     (system prompts)               |
|        +-->  domain/      (core types)                   |
|        +-->  config/      (TOML + env var config)        |
|        +-->  backend/     (LLM backend discovery)        |
+----------------------------------------------------------+
```

## Package Overview

| Package | Purpose |
|---------|---------|
| `config` | Centralized config types, koanf loading (TOML + env vars), `WriteTo()` for persistence |
| `api` | Chi router, HTTP handlers, OpenAI-compatible DTOs |
| `admin` | Admin dashboard UI: backends CRUD, config editor, crawler control, documents, agent chat |
| `agent` | Session-aware tool-calling LLM loop (configurable max iterations) with semantic_search tool; single entry point for all conversations |
| `rag` | Retrieval service (retrieve + build context + ingest); takes embedder only |
| `llm` | openai-go SDK wrapper for embeddings/chat/streaming/tool-calling |
| `store` | SQLite document/chunk storage, cosine similarity vector search, backend + crawl source CRUD |
| `crawler` | BFS web crawler: 3-stage pipeline (crawl -> parse -> embed), go-rod for JS pages, go-readability for content extraction, PDF support |
| `objectstore` | S3-compatible object storage client (RustFS) |
| `chunking` | StructureAwareChunker (heading-boundary splitting), RecursiveTokenChunker (fallback), RecursiveCharacterChunker (char-based fallback) |
| `tokencount` | Token counting: LlamaCppCounter (HTTP /tokenize), TiktokenCounter (tiktoken-go), EstimateCounter (len/4 fallback) |
| `session` | Conversation state with TTL expiration |
| `domain` | Core types: Document, Chunk, SourceReference, Language, Category |
| `backend` | LLM backend discovery (llama.cpp props, model listing) |
| `prompts` | `text/template` system prompt rendering with data injection (doc/chunk counts) |
| `server` | Application wiring (backend resolution, client construction, hot-rebuild on config save), HTTP server lifecycle |

## Data Flow

### RAG Retrieval

```
User query
    |
    v
[Embed query] --> [Vector search (cosine similarity)] --> [Top-K chunks]
    |
    v
[Build context] --> [Formatted chunk text with token budget]
```

The pipeline is a pure retrieval service — it takes an `embedder` only. Chat generation is handled by the agent, which calls `semantic_search` as a tool.

### Crawl Pipeline

```
Seed URLs --> [BFS crawl] --> [Fetch (HTTP/Browser)] --> [Store raw in RustFS]
                                    |
                                    v
                            [Parse (readability/pdftotext)]
                                    |
                                    v
                            [Chunk + Embed] --> [SQLite (docs + chunks + embeddings)]
```

Three-stage concurrent pipeline using Go channels. Each source runs independent BFS with its own rate limiter.

### Agent Loop

The agent is the single entry point for all conversations. It owns the system prompt, manages sessions, and collects citations from tool calls.

`Tool` is an interface (`Name`, `Description`, `Parameters`, `Execute`). Each concrete tool (e.g. `SemanticSearchTool`) is its own type. `Execute` returns a `ToolResult` carrying the LLM-facing text and optional `[]Citation` (tool-agnostic `{Title, URL}` references used to render links in the UI and in the REST response).

```
User message --> [Load session history] --> [Chat with tools] --> [Tool call?]
                                                  ^                     |
                                                  |                     v
                                                  +---- [Execute tool]
                                                        [Aggregate ToolResult.Citations]

                                           (max_iterations configurable, default 5)
```

Methods: `Chat()` (session-managed), `ChatStream()` (session + streaming), `ChatPassthrough()` (stateless, client provides history).
Live reconfiguration: `rebuildClients()` constructs a fresh `Agent` on config save and installs it via `Handler.SetAgent()` on the API and admin handlers — individual fields are not mutated at runtime. Agent construction (search-tool wiring, vision getter) is centralised in `Server.buildAgent()` so the initial-setup and rebuild paths cannot drift.

## Key Design Patterns

- **Backends as named endpoints**: `BackendConfig` has `IsLlamaCpp` flag; stage config (`[embedding]`, `[agent]`) references backend by name + specifies model
- **Agent as single chat entry point**: API handler delegates all chat to the agent; RAG pipeline is a pure retrieval service with no chat client
- **`server` package as compositor**: `server.Server.resolveEmbedClient()` / `resolveChatClient()` (in `internal/server/wiring.go`) resolve a backend by name, merge with stage config, and build `llm.EmbedClient` or `llm.ChatClient`. `resolveCounter()` creates the appropriate token counter. `probeLlamaCppBackends()` queries `/props` to auto-detect context size and embedding dimension. `rebuildClients()` (in `rebuild.go`) hot-swaps all clients after a config save in the admin UI
- **Markdown throughout**: Parser emits markdown, stored as markdown, embedded as markdown, served to LLM as markdown
- **Content archival**: Raw content stored in RustFS, enabling re-processing (new chunk size, new embedding model) without re-crawling

## Routes

- **Health**: `GET /health`, `GET /v1/health`
- **API**: `/v1/chat/completions` (session mode or passthrough mode), `/v1/embeddings`, `/v1/rag/ingest`, `GET /v1/rag/documents`, `DELETE /v1/rag/documents/{id}`, `GET /v1/artifacts/{token}.png` (image artifact streaming for vision)
- **Admin**: `/admin/*` (dashboard, backends, config, crawler, documents, agent chat)
- Admin UI: templ -> `_templ.go`, Tailwind CSS v4 standalone CLI, HTMX + SSE

## Storage

```
RustFS (S3)             SQLite
+----------------+      +--------------------+
| Raw HTML       |      | documents          |
| Raw PDFs       |      | chunks             |
| crawl/{type}   |      | chunk_embeddings   |
|   /{sha256}    |      | document_artifacts |
| Page images    |      | backends           |
| crawl/img/     |      | crawl_sources      |
|   {sha256}/    |      | crawl_pages        |
|   {ref}.png    |      | crawl_runs         |
+----------------+      +--------------------+
```

**Why both?** RustFS preserves raw content for re-processing. SQLite holds structured data, metadata, and vectors (brute-force cosine similarity in Go — fast enough up to ~10K chunks, see [ADR-002](decisions/002-brute-force-cosine.md)).

## Multimodal artifacts

Image artifacts (PDF pages via `pdftoppm`, HTML full-page screenshots via go-rod) are rendered during crawl and stored under `crawl/img/{url_sha256}/{ref}.png` in RustFS, with a `document_artifacts` row linking them to the parent document by `doc_id`. The parser embeds inline `[[image:ref]]` markers in chunk text; embeddings strip them in-flight, retrieval scans them to attach `image_url` parts to a synthetic user message when the configured backend is vision-capable. See [ADR-006](decisions/006-multimodal-artifacts.md) for the full design.
