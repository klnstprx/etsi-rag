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
|        +-->  sources/     (crawl-source registry)        |
|        +-->  backend/     (LLM backend discovery)        |
|        +-->  diag/        (health-issue registry)        |
|        +-->  markers/     (inline image-ref syntax)      |
+----------------------------------------------------------+
```

## Package Overview

| Package | Purpose |
|---------|---------|
| `config` | Centralized config types, koanf loading (TOML + env vars), `WriteTo()` for persistence |
| `api` | Chi router, HTTP handlers, OpenAI-compatible DTOs |
| `admin` | Admin dashboard UI: backends CRUD, config editor, crawler control, documents, agent chat |
| `agent` | Session-aware tool-calling LLM loop (configurable max iterations) with `semantic_search` and `read_document` tools; single entry point for all conversations |
| `rag` | Retrieval service (`Retriever.Retrieve` + `BuildContext` + `SourcesFromRetrieval`, plus RRF and reranker glue); takes store + shared `*llm.EmbedderHandle` + counter, plus optional reranker — no chat client, no ingest |
| `ingest` | Indexing service (`Ingester.Ingest` + `HealthCheck`); chunk → embed → store. Crawler and `POST /v1/rag/ingest` both go through here. Owns the `ErrChunking` sentinel |
| `llm` | openai-go SDK wrapper for embeddings/chat/streaming/tool-calling; cross-encoder reranker client (`rerank.go`, llama.cpp `/v1/rerank`) |
| `store` | SQLite document/chunk storage, cosine similarity vector search, FTS5 BM25 keyword search, crawl-page/run + artifact CRUD (backends and crawl sources live in TOML, not SQLite) |
| `crawler` | BFS web crawler: 3-stage pipeline (crawl -> parse -> embed), go-rod for JS pages, go-readability for content extraction, PDF support |
| `objectstore` | S3-compatible object storage client (RustFS) |
| `chunking` | StructureAwareChunker (heading-boundary splitting; falls back to in-package separator + token-range splitter for oversized sections and heading-free documents) |
| `tokencount` | Token counting: LlamaCppCounter (HTTP /tokenize), TiktokenCounter (tiktoken-go) |
| `session` | Conversation state with TTL expiration |
| `domain` | Core types: Document, Chunk, SourceReference, Language, CrawlSource, Artifact |
| `sources` | In-memory `Registry` over `config/sources.toml`; admin edits flow through `Registry.Replace` and the crawler / `list_sources`-style consumers read the snapshot on demand |
| `backend` | LLM backend discovery (llama.cpp `/props`, model listing) |
| `diag` | Startup + runtime issue registry surfaced on the admin dashboard (e.g. object-store init failure, missing sources file); supports per-issue `Recheck` callbacks |
| `prompts` | `text/template` system prompt rendering with data injection (doc/chunk counts) |
| `markers` | `[[image:ref]]` syntax + `ExtractBoundaries` helper used by the parser→chunker handoff to map each chunk's source span to image artifact refs (stored separately on `chunks.image_refs`, not in chunk text) |
| `server` | Application wiring (backend resolution, client construction, hot-rebuild on config save), HTTP server lifecycle |

## Data Flow

### RAG Retrieval

```
User query
    |
    +-- [retrieval_mode = "dense"]   --> [Embed] --> [Cosine top-K_pool]   -+
    |                                                                      |
    +-- [retrieval_mode = "hybrid"]  --> [Embed] --> [Cosine top-K_each]   -+
    |                                                                      |
    |                                       [BM25 top-K_each] (FTS5) ------+--> [RRF merge] --> [pool]
    |                                                                                              |
    |                                                                                              v
    |                                                                                   [Reranker (optional)]
    |                                                                                              |
    |                                                                                              v
    v                                                                                          [Top-K]
[Build context] --> [Formatted chunk text with token budget]
```

The `rag` package is a pure retrieval service — `Retriever` takes a shared `*llm.EmbedderHandle` and (optionally) a reranker. Ingestion (chunk + embed + store) lives in the separate `ingest` package; both halves share the same embedder handle so the "one embedding model per corpus" invariant is enforced by construction (a config-reload swaps the embedder in one place and both sides observe the new client on next Get). Chat generation is handled by the agent, which calls `semantic_search` as a tool.

`retrieval_mode` is selected via `[retrieval]` in `config.toml`. Default is `"dense"`. `"hybrid"` adds a BM25 leg over an SQLite FTS5 index of `chunks.content` and merges with cosine via reciprocal-rank fusion (`k=60`). When `[reranker].enabled = true`, a cross-encoder (llama.cpp `/v1/rerank`) re-scores the candidate pool to produce the final top-K (soft fall-through if the backend errors). For formulas, parameter choices, and citations see [retrieval.md](retrieval.md); see [ADR-008](decisions/008-hybrid-retrieval.md) for hybrid and [ADR-009](decisions/009-reranker.md) for the reranker. The agent's `semantic_search` tool sees the same `Retrieve` contract regardless of mode.

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

`Tool` is an interface (`Name`, `Description`, `Parameters`, `Execute`). Each concrete tool is its own type. `Execute` returns a `ToolResult` carrying the LLM-facing text and optional `[]Citation` (tool-agnostic `{Title, URL}` references used to render links in the UI and in the REST response).

| Tool | Purpose |
|------|---------|
| `semantic_search` | Top-K dense / hybrid retrieval (optional `top_k` override). Reads each retrieved chunk's `ImageRefs` to attach the matching artifacts when the backend supports vision. A previous `source_keys` / `tags` scoping path and the companion `list_sources` tool were reverted on 2026-05-09 after an accuracy regression on small models. |
| `read_document` | Fetch a full document by URL (for follow-up after a too-narrow snippet). Optional `query` runs a doc-scoped dense search and returns the top-K chunks within that document. Reuses `documents.content` and the per-doc chunk vectors — RustFS is not on the read path. |

```
User message --> [Load session history] --> [Chat with tools] --> [Tool call?]
                                                  ^                     |
                                                  |                     v
                                                  +---- [Execute tool]
                                                        [Aggregate ToolResult.Citations]
                                                        [If image refs + vision-capable model:
                                                         inject synthetic user message with
                                                         image_url parts]

                                           (max_iterations configurable, default 5)
```

Methods: `Chat()` (session-managed), `ChatStream()` (session + streaming), `ChatPassthrough()` (stateless, client provides history).
Vision: when `[agent.vision].enabled = true` and `public_base_url` is set, the agent reads each retrieved chunk's `ImageRefs` (stamped at chunk time from the parser's page markers), resolves them to artifact URLs, and injects a synthetic user message carrying `image_url` parts on the next turn (see [ADR-006](decisions/006-multimodal-artifacts.md)).
Live reconfiguration: `rebuildClients()` constructs a fresh `Agent` on config save and routes it through the shared `*agent.Handle` — one `handle.Set(newAgent)` call propagates to both the API and admin handlers via `handle.Get()`, eliminating parallel-fanout fragility. The embedder follows the same pattern (`*llm.EmbedderHandle` shared by retriever and ingester). Agent construction (search-tool wiring, vision getter, reranker wiring) is centralised in `Server.buildAgent()` so the initial-setup and rebuild paths cannot drift.

## Key Design Patterns

- **Backends as named endpoints**: `BackendConfig` has `IsLlamaCpp` flag; stage config (`[embedding]`, `[agent]`) references backend by name + specifies model
- **Agent as single chat entry point**: API handler delegates all chat to the agent; RAG pipeline is a pure retrieval service with no chat client
- **`server` package as compositor**: `server.Server.resolveEmbedClient()` / `resolveChatClient()` (in `internal/server/wiring.go`) resolve a backend by name, merge with stage config, and build `llm.EmbedClient` or `llm.ChatClient`. `resolveCounter()` creates the appropriate token counter. `probeLlamaCppBackends()` queries `/props` to auto-detect context size and embedding dimension. `rebuildClients()` (in `rebuild.go`) hot-swaps all clients after a config save in the admin UI
- **Markdown throughout**: Parser emits markdown, stored as markdown, embedded as markdown, served to LLM as markdown
- **Content archival**: Raw content stored in RustFS, enabling re-processing (new chunk size, new embedding model) without re-crawling

## Routes

- **Health**: `GET /health`, `GET /v1/health`
- **API**: `/v1/chat/completions` (session mode or passthrough mode), `/v1/embeddings`, `/v1/rag/ingest`, `GET /v1/rag/documents`, `DELETE /v1/rag/documents/{id}`, `POST /v1/rag/query` (retrieval-layer introspection for eval — returns ranked chunks with document_url, used to compute Recall@k / MRR), `GET /v1/artifacts/{token}.png` (image artifact streaming for vision)
- **Admin**: `/admin/*` (dashboard, backends, config, crawler, documents, agent chat)
- Admin UI: templ -> `_templ.go`, Tailwind CSS v4 standalone CLI, HTMX + SSE

## Storage

```
RustFS (S3)             SQLite                            config/
+----------------+      +-------------------------------+ +-------------+
| Raw HTML       |      | documents                     | | config.toml |
| Raw PDFs       |      | chunks (+ image_refs column)  | | sources.toml|
| crawl/{type}   |      | chunks_fts  (FTS5 virtual)    | +-------------+
|   /{sha256}    |      | chunk_embeddings              |
| Page images    |      | document_artifacts            |
| crawl/img/     |      | crawl_pages                   |
|   {sha256}/    |      | crawl_runs                    |
|   {ref}.png    |      |                               |
+----------------+      +-------------------------------+
```

Backends (`[[backends]]`) and crawl sources (`[[sources]]`) live in the TOML files,
not in SQLite — admin CRUD writes the file and updates the in-memory snapshot.

Crawl sources live in `config/sources.toml`, not the database. `crawl_pages.source_key` and `crawl_runs.source_key` reference the stable slug from each `[[sources]]` block. Admin CRUD writes back to `sources.toml` and calls `sources.Registry.Replace` so the crawler (and any other consumer) picks up the new snapshot on its next read — no server restart required.

**Why both?** RustFS preserves raw content for re-processing. SQLite holds structured data, metadata, and vectors. Brute-force cosine similarity in Go, with an in-memory embedding cache warmed at startup so retrieval avoids re-loading 16 K BLOBs from SQLite per query (~37 ms cached vs ~370 ms uncached at the current corpus); see [ADR-002](decisions/002-brute-force-cosine.md).

## Multimodal artifacts

Image artifacts (PDF pages via `pdftoppm`, HTML full-page screenshots via go-rod) are rendered during crawl and stored under `crawl/img/{url_sha256}/{ref}.png` in RustFS, with a `document_artifacts` row linking them to the parent document by `doc_id`. The parser emits one `[[image:ref]]` page-boundary line per page top; the chunker consumes those markers as a parser→chunker protocol, stamps each emitted chunk with the refs whose page range overlaps the chunk's source section (stored on `chunks.image_refs`), and strips the markers from final chunk content. Retrieval reads `ImageRefs` directly to attach `image_url` parts to a synthetic user message when the configured backend is vision-capable. See [ADR-006](decisions/006-multimodal-artifacts.md) for the full design.
