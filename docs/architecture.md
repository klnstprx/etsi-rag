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
|  cmd/server/main.go                                      |
|        |                                                 |
|        +-->  api/         (OpenAI-compatible + RAG API)  |
|        +-->  admin/       (templ + Tailwind + HTMX UI)   |
|        +-->  agent/       (tool-calling LLM loop)        |
|        +-->  rag/         (retrieve -> context -> gen)   |
|        +-->  crawler/     (BFS web crawler, 3-stage)     |
|        +-->  llm/         (openai-go SDK wrapper)        |
|        +-->  store/       (SQLite + cosine similarity)   |
|        +-->  objectstore/ (S3/RustFS client)             |
|        +-->  session/     (conversation state + TTL)     |
|        +-->  chunking/    (token-aware text splitting)   |
|        +-->  tokencount/  (token counting abstraction)   |
|        +-->  prompts/     (system prompts)               |
|        +-->  domain/      (core types)                   |
|        +-->  data/        (data source abstraction)      |
+----------------------------------------------------------+
```

## Package Overview

| Package | Purpose |
|---------|---------|
| `config` | Centralized config types, koanf loading (TOML + env vars), `WriteTo()` for persistence |
| `api` | Chi router, HTTP handlers, OpenAI-compatible DTOs |
| `admin` | Admin dashboard UI: backends CRUD, config editor, crawler control, documents, agent chat |
| `agent` | Tool-calling LLM loop (max 5 iterations) with rag_search + list_documents tools |
| `rag` | Pipeline orchestration (retrieve -> build context -> generate); takes separate embedder + chatter |
| `llm` | openai-go SDK wrapper for embeddings/chat/streaming/tool-calling |
| `store` | SQLite document/chunk storage, cosine similarity vector search, backend + crawl source CRUD |
| `crawler` | BFS web crawler: 3-stage pipeline (crawl -> parse -> embed), go-rod for JS pages, go-readability for content extraction, PDF support |
| `objectstore` | S3-compatible object storage client (RustFS) |
| `chunking` | StructureAwareChunker (heading-boundary splitting), RecursiveTokenChunker (fallback), RecursiveCharacterChunker (char-based fallback) |
| `tokencount` | Token counting: LlamaCppCounter (HTTP /tokenize), TiktokenCounter (tiktoken-go), EstimateCounter (len/4 fallback) |
| `session` | Conversation state with TTL expiration |
| `domain` | Core types: Document, Chunk, Backend, CrawlSource, Language, Category |
| `data` | Data source abstraction + mock |
| `prompts` | System prompt templates |

## Data Flow

### RAG Pipeline

```
User query
    |
    v
[Embed query] --> [Vector search (cosine similarity)] --> [Top-K chunks]
    |
    v
[Build context] --> [System prompt + retrieved chunks + conversation history]
    |
    v
[Chat completion] --> [Streamed response to user]
```

The pipeline takes separate `embedder` and `chatter` LLM clients, allowing different backends/models for embedding vs. chat.

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

```
User message --> [Chat with tools] --> [Tool call?]
                      ^                     |
                      |                     v
                      +---- [Execute tool (rag_search / list_documents)]

                (max 5 iterations)
```

## Key Design Patterns

- **Backends as named endpoints**: `BackendConfig` has `IsLlamaCpp` flag; stage config (`[embedding]`, `[chat]`, `[agent]`) references backend by name + specifies model
- **Split LLM dependencies**: Pipeline takes `embedder` + `chatter`; API handler takes `embeddingClient` + `chatClient`
- **`main.go` as compositor**: `clientForStage()` resolves backend by name, merges with stage config, builds `llm.Client`. `counterForBackend()` creates the appropriate token counter
- **Markdown throughout**: Parser emits markdown, stored as markdown, embedded as markdown, served to LLM as markdown
- **Content archival**: Raw content stored in RustFS, enabling re-processing (new chunk size, new embedding model) without re-crawling

## Routes

- **API**: `/v1/chat/completions`, `/v1/embeddings`, `/v1/rag/*`
- **Admin**: `/admin/*` (dashboard, backends, config, crawler, documents, agent chat)
- Admin UI: templ -> `_templ.go`, Tailwind CSS v4 standalone CLI, HTMX + SSE

## Storage

```
RustFS (S3)          SQLite
+--------------+     +-------------------+
| Raw HTML     |     | documents         |
| Raw PDFs     |     | chunks            |
| crawl/{type} |     | chunk_embeddings  |
|   /{sha256}  |     | backends          |
+--------------+     | crawl_sources     |
                     | crawl_pages       |
                     | crawl_runs        |
                     +-------------------+
```

**Why both?** RustFS preserves raw content for re-processing. SQLite holds structured data, metadata, and vectors (brute-force cosine similarity in Go — fast enough up to ~10K chunks, see [ADR-002](decisions/002-brute-force-cosine.md)).
