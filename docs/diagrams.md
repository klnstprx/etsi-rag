# Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%
graph TD
    Main["cmd/server/main.go<br/>(entry)"] --> Server["internal/server/<br/>(compositor + lifecycle)"]

    Server --> API
    Server --> Admin

    subgraph HTTP["HTTP Layer"]
        API["api/<br/>OpenAI-compatible + RAG"]
        Admin["admin/<br/>templ + Tailwind + HTMX"]
    end

    API --> Agent
    Admin --> Agent
    Admin --> Crawler

    subgraph Core["Core Logic"]
        Agent["agent/<br/>Tool-calling LLM loop"]
        RAG["rag/<br/>Retrieval service"]
        Crawler["crawler/<br/>BFS 3-stage pipeline"]
    end

    Agent --> RAG
    Crawler --> RAG

    subgraph Infra["Infrastructure"]
        LLMPkg["llm/<br/>openai-go SDK wrapper"]
        Store["store/<br/>SQLite + cosine similarity"]
        ObjStore["objectstore/<br/>S3 client"]
        Chunking["chunking/<br/>Structure-aware splitting"]
    end

    RAG --> Store
    RAG --> Chunking
    Agent --> LLMPkg
    RAG --> LLMPkg
    Crawler --> ObjStore

    subgraph External["External Services"]
        LLMBackend["LLM Backend<br/>(llama-server)"]
        SQLite[("SQLite")]
        S3[("RustFS / S3")]
        Chrome["Headless Chrome"]
    end

    LLMPkg --> LLMBackend
    Store --> SQLite
    ObjStore --> S3
    Crawler --> Chrome

    subgraph Shared["Shared (types &amp; utilities, used by all layers)"]
        Config["config/"] ~~~ Domain["domain/"] ~~~ Prompts["prompts/"] ~~~ Markers["markers/"] ~~~ Backend["backend/"] ~~~ Session["session/"] ~~~ TokenCount["tokencount/"]
    end

    External ~~~ Shared
```

## 2. Package Dependency Graph

```mermaid
graph TD
    main["cmd/server/main.go"]
    server["internal/server/"]

    main --> server
    server --> api & admin

    subgraph Handlers
        api
        admin
    end

    api --> agent & rag & llm & session & store & objectstore
    admin --> agent & crawler & rag & config

    subgraph Core
        agent
        rag
        crawler
    end

    agent --> llm & rag
    rag --> llm & store & chunking & tokencount
    crawler --> store & objectstore & rag

    subgraph Infra
        llm
        store
        objectstore
        session
        chunking
    end

    chunking --> tokencount

    subgraph Shared
        config
        domain
        tokencount
        prompts
        markers
        backend
    end

    store --> domain
    chunking --> config & domain
    rag --> domain & config & markers
    api --> domain & tokencount
    crawler --> domain & markers
    server --> backend
    agent --> markers
```

## 3. RAG Pipeline

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion Flow"]
        direction TB
        Doc["Document<br/>(markdown + [[image:ref]] markers)"] --> Chunk["Chunk<br/>(StructureAwareChunker:<br/>strips markers, stamps<br/>chunks.image_refs)"]
        Chunk --> Embed["Embed chunks<br/>(marker-free content)"]
        Embed --> StoreDoc["Store document +<br/>chunks (+ image_refs) +<br/>embeddings + chunks_fts (FTS5)"]
        Doc -.->|"page renders / screenshots"| Artifacts["document_artifacts<br/>(RustFS PNGs)"]
    end

    subgraph Query["Query Flow (retrieval-only — agent owns chat)"]
        direction TB
        Q["User query"] --> Agent["agent.Chat<br/>(LLM + tools loop)"]
        Agent -->|"semantic_search tool call"| EmbedQ["Embed tool query"]
        EmbedQ --> Mode{"retrieval_mode"}
        Mode -->|"dense"| Cosine["Cosine top-K_pool"]
        Mode -->|"hybrid"| HybridBox
        subgraph HybridBox["Hybrid"]
            direction LR
            Cos2["Cosine top-K_each"] --> RRF["RRF merge<br/>(k=60)"]
            BM25["BM25 top-K_each<br/>(chunks_fts)"] --> RRF
        end
        Cosine --> Pool["Candidate pool"]
        HybridBox --> Pool
        Pool --> Rerank{"reranker.enabled?"}
        Rerank -->|"yes"| Cross["Cross-encoder<br/>(/v1/rerank)"]
        Rerank -->|"no / soft fall-through"| TopK
        Cross --> TopK["Top-K chunks"]
        TopK --> BuildCtx["Build context<br/>(chunks + metadata + image refs)"]
        BuildCtx -->|"ToolResult.Text + Citations + Images"| Agent
        Agent --> Response["Streamed response<br/>(text + citations)"]
    end

    StoreDoc -.->|"stored embeddings"| Cosine
    StoreDoc -.->|"stored embeddings"| Cos2
    StoreDoc -.->|"FTS5 index"| BM25
    Artifacts -.->|"resolved by markers"| BuildCtx
```

## 4. Crawler Pipeline (3-Stage)

```mermaid
flowchart TD
    Seeds["Seed URLs"] --> BFS["BFS traversal"]
    BFS --> Fetch["Fetch (HTTP / go-rod)"]
    Fetch --> RawStore["Store raw in RustFS"]
    Fetch --> Links["Extract links"]
    Links -->|"enqueue"| BFS

    Fetch -->|"parseCh"| Parse

    subgraph Parse["Stage 2: Parse"]
        direction LR
        HTML["HTML"] --> Readability["go-readability"]
        PDFIn["PDF"] --> PDFText["pdftotext (poppler)"]
        Readability --> MD["Markdown<br/>(with [[image:ref]] markers)"]
        PDFText --> MD
        HTML -.->|"go-rod screenshot"| Screenshot["HTML page PNG"]
        PDFIn -.->|"pdftoppm"| PDFPages["PDF page PNGs"]
        Screenshot --> ArtUpload["Upload to RustFS<br/>crawl/img/{sha}/{ref}.png"]
        PDFPages --> ArtUpload
    end

    Parse -->|"embedCh"| HashCheck
    Parse -.->|"document_artifacts row<br/>(doc_id, ref, public_token)"| ArtTable[("document_artifacts")]

    HashCheck{"Content changed?"}
    HashCheck -->|"Yes"| Ingest["RAG Ingest<br/>(chunk + embed + store + FTS5)"]
    HashCheck -->|"No"| Skip["Skip (update timestamp)"]
```

## 5. Agent Tool-Calling Loop

```mermaid
flowchart TB
    Start["User message"] --> Build["Build messages<br/>(system + history + user)"]
    Build --> LLMCall["Chat with tools<br/>(LLM backend)"]
    LLMCall --> Check{"Tool calls<br/>in response?"}

    Check -->|"No"| Final["Return text response<br/>+ aggregated citations"]
    Check -->|"Yes"| Execute["Execute tool(s)"]

    Execute --> SemanticSearch["semantic_search<br/>(retrieve + build context)"]
    Execute --> ReadDocument["read_document<br/>(whole-doc fetch, or<br/>doc-scoped dense search<br/>when query is set)"]
    SemanticSearch --> Result["ToolResult<br/>{Text, Citations, Images}"]
    ReadDocument --> Result

    Result --> AggCite["Aggregate Citations<br/>(into ChatResult/ToolEvent)"]
    AggCite --> Append["Append tool message<br/>to history"]

    Append --> Vision{"Images present<br/>and vision enabled?"}
    Vision -->|"Yes"| Inject["Inject synthetic user message<br/>with image_url parts<br/>(public artifact URLs)"]
    Vision -->|"No"| Counter
    Inject --> Counter

    Counter{"Iterations<br/>< max_iterations?"}
    Counter -->|"Yes"| LLMCall
    Counter -->|"No"| ErrorBox["Error: max iterations<br/>([agent].max_iterations, default 5)"]
```

## 6. Configuration & Client Resolution

```mermaid
flowchart TD
    subgraph Loading["Config Loading"]
        direction LR
        Default["default.toml"] & EnvToml["{RUN_MODE}.toml"] & EnvVars["APP__* env vars"] --> Koanf
        Koanf --> Config["Config struct"]
    end

    Config --> Stage["Stage config<br/>([embedding] / [agent] / [reranker])"]
    Stage --> Resolve["server.resolveEmbedClient / resolveChatClient / resolveReranker<br/>(GetBackend by name + merge stage cfg)"]
    Resolve --> Backend["BackendConfig"]

    Backend --> EmbedClient["llm.EmbedClient"]
    Backend --> ChatClient["llm.ChatClient"]
    Backend --> RerankClient["llm.RerankClient<br/>(llama.cpp /v1/rerank)"]
    Backend --> Counter{"IsLlamaCpp?"}
    Counter -->|"Yes"| LlamaCounter["LlamaCppCounter (HTTP /tokenize)"]
    Counter -->|"No"| Tiktoken["TiktokenCounter (tiktoken-go)"]

    Stage -.->|"[agent.vision].enabled<br/>+ public_base_url"| VisionFlag["Agent vision<br/>(synthetic image_url messages)"]
```

## 7. Request Routing

```mermaid
flowchart TB
    HTTP["HTTP Request"] --> Router["Chi Router"]

    Router --> Health["GET /health"]
    Router --> V1["/v1/*"]
    Router --> AdminRoutes["/admin/*"]

    V1 --> V1Health["GET /v1/health"]
    V1 --> ChatComp["POST /v1/chat/completions"]
    V1 --> Embeddings["POST /v1/embeddings"]
    V1 --> Artifacts["GET /v1/artifacts/{token}<br/>(image artifact stream)"]
    V1 --> RAGRoutes["/v1/rag/*"]

    RAGRoutes --> Ingest["POST /v1/rag/ingest"]
    RAGRoutes --> ListDocs["GET /v1/rag/documents"]
    RAGRoutes --> DeleteDoc["DELETE /v1/rag/documents/{id}"]
    RAGRoutes --> Query["POST /v1/rag/query<br/>(retrieval introspection for eval)"]

    AdminRoutes --> Dashboard["GET /admin/"]
    AdminRoutes --> Backends["GET/POST /admin/backends"]
    AdminRoutes --> ConfigPage["GET/POST /admin/config"]
    AdminRoutes --> CrawlerPage["GET/POST /admin/crawler"]
    AdminRoutes --> Documents["GET /admin/documents"]
    AdminRoutes --> AgentChat["GET /admin/chat"]
```

## 8. Storage Schema

```mermaid
erDiagram
    documents {
        text id PK
        text title
        text content
        text url
        text language "es | en"
        datetime created_at
    }

    chunks {
        int id PK "AUTOINCREMENT"
        text doc_id FK
        text content
        int chunk_index
        text section_title
        text image_refs "comma-separated PNG refs (p1,p2,full,…)"
    }

    chunks_fts {
        text content "FTS5 virtual table"
    }

    chunk_embeddings {
        int chunk_id PK
        blob embedding
    }

    crawl_pages {
        int id PK "AUTOINCREMENT"
        text source_key "slug from sources.toml"
        text url UK
        text content_type
        text title
        text content_hash
        text object_store_key
        text doc_id FK
        text status
        text error_message
        datetime last_crawled_at
        datetime last_changed_at
    }

    crawl_runs {
        int id PK "AUTOINCREMENT"
        text source_key "slug; nullable for run-all"
        text status
        int pages_discovered
        int pages_fetched
        int pages_changed
        int pages_failed
        datetime started_at
        datetime finished_at
    }

    document_artifacts {
        int id PK "AUTOINCREMENT"
        text doc_id FK
        text ref
        text kind
        text object_store_key
        text mime_type
        int width
        int height
        text public_token UK
    }

    documents ||--o{ chunks : "has"
    chunks ||--|| chunk_embeddings : "has"
    chunks ||--|| chunks_fts : "indexed via triggers"
    documents ||--o{ document_artifacts : "has"
    crawl_pages }o--o| documents : "references"
```

Backends and crawl sources are **not stored in SQLite** — they live as `[[backends]]` in `config.toml` and `[[sources]]` in `config/sources.toml` respectively. Both round-trip through admin UI CRUD via `config.WriteTo()` and `config.WriteSources()`. `crawl_pages.source_key` and `crawl_runs.source_key` reference the slug from each `[[sources]]` block, validated against the in-memory snapshot at runtime rather than by SQL FK. The `chunks_fts` virtual table (FTS5 over `chunks.content`, `unicode61 remove_diacritics 2` tokenizer) is kept in sync with `chunks` via INSERT/UPDATE/DELETE triggers and backs the BM25 leg of hybrid retrieval.
