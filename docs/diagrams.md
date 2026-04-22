# Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TD
    Main["cmd/server/main.go<br/>(entry point)"]
    Server["internal/server/<br/>(compositor + lifecycle)"]

    Main --> Server
    Server --> API & Admin

    subgraph HTTP["HTTP Layer"]
        API["api/<br/>OpenAI-compatible + RAG"]
        Admin["admin/<br/>templ + Tailwind + HTMX"]
    end

    API --> RAG
    Admin --> Agent & Crawler

    subgraph Core["Core Logic"]
        Agent["agent/<br/>Tool-calling LLM loop"]
        RAG["rag/<br/>Pipeline orchestration"]
        Crawler["crawler/<br/>BFS 3-stage pipeline"]
    end

    Agent --> RAG
    Crawler --> RAG

    RAG --> LLMPkg & Store & Chunking
    Agent --> LLMPkg
    Crawler --> ObjStore

    subgraph Infra["Infrastructure"]
        LLMPkg["llm/<br/>openai-go SDK wrapper"]
        Store["store/<br/>SQLite + cosine similarity"]
        ObjStore["objectstore/<br/>S3 client"]
        Session["session/<br/>Conversation state + TTL"]
        Chunking["chunking/<br/>Structure-aware splitting"]
    end

    Chunking --> TokenCount["tokencount/<br/>Token counting"]

    LLMPkg -->|"openai-go SDK"| LLM
    Store --> SQLite
    ObjStore -->|"aws-sdk-go-v2"| S3
    Crawler -->|"go-rod"| Chrome

    subgraph External["External Services"]
        LLM["LLM Backend<br/>(llama-server / OpenAI-compatible)"]
        S3[("RustFS / S3")]
        Chrome["Headless Chrome"]
    end

    SQLite[("SQLite")]

    subgraph Shared
        Config["config/"] ~~~ Domain["domain/"] ~~~ Prompts["prompts/"]
    end
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

    api --> rag & llm & session
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
    end

    store --> domain
    chunking --> config & domain
    rag --> domain & config
    api --> domain & tokencount
    crawler --> domain
```

## 3. RAG Pipeline

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion Flow"]
        direction TB
        Doc["Document"] --> Chunk["Chunk<br/>(StructureAwareChunker)"]
        Chunk --> Embed["Embed chunks<br/>(embedding model)"]
        Embed --> StoreDoc["Store document +<br/>chunks + embeddings<br/>(SQLite)"]
    end

    subgraph Query["Query Flow"]
        direction TB
        Q["User query"] --> EmbedQ["Embed query"]
        EmbedQ --> Search["Vector search<br/>(cosine similarity)"]
        Search --> TopK["Top-K chunks"]
        TopK --> BuildCtx["Build context<br/>(chunks + metadata)"]
        BuildCtx --> Messages["System prompt +<br/>context + history"]
        Messages --> Chat["Chat completion<br/>(chat model)"]
        Chat --> Response["Streamed response"]
    end

    StoreDoc -.->|"stored embeddings"| Search
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
        Readability --> MD["Markdown"]
        PDFText --> MD
    end

    Parse -->|"embedCh"| HashCheck

    HashCheck{"Content changed?"}
    HashCheck -->|"Yes"| Ingest["RAG Ingest<br/>(chunk + embed + store)"]
    HashCheck -->|"No"| Skip["Skip (update timestamp)"]
```

## 5. Agent Tool-Calling Loop

```mermaid
flowchart TB
    Start["User message"] --> Build["Build messages<br/>(system + user)"]
    Build --> LLMCall["Chat with tools<br/>(LLM backend)"]
    LLMCall --> Check{"Tool calls<br/>in response?"}

    Check -->|"No"| Final["Return text response"]
    Check -->|"Yes"| Execute["Execute tool(s)"]

    Execute --> SemanticSearch["semantic_search<br/>(vector search + context)"]
    Execute --> ListDocs["list_documents<br/>(document listing)"]

    SemanticSearch --> Append["Append tool results<br/>to messages"]
    ListDocs --> Append

    Append --> Counter{"Iterations<br/>< 5?"}
    Counter -->|"Yes"| LLMCall
    Counter -->|"No"| Error["Error: max iterations"]
```

## 6. Configuration & Client Resolution

```mermaid
flowchart TD
    subgraph Loading["Config Loading"]
        direction LR
        Default["default.toml"] & EnvToml["{RUN_MODE}.toml"] & EnvVars["APP__* env vars"] --> Koanf
        Koanf --> Config["Config struct"]
    end

    Config --> Stage["Stage config ([embedding] / [agent])"]
    Stage --> Resolve["server.resolveEmbedClient / resolveChatClient<br/>(GetBackend by name + merge stage cfg)"]
    Resolve --> Backend["BackendConfig"]

    Backend --> EmbedClient["llm.EmbedClient"]
    Backend --> ChatClient["llm.ChatClient"]
    Backend --> Counter{"IsLlamaCpp?"}
    Counter -->|"Yes"| LlamaCounter["LlamaCppCounter (HTTP /tokenize)"]
    Counter -->|"No"| Tiktoken["TiktokenCounter (tiktoken-go)"]
```

## 7. Request Routing

```mermaid
flowchart TB
    HTTP["HTTP Request"] --> Router["Chi Router"]

    Router --> Health["/health"]
    Router --> V1["/v1/*"]
    Router --> AdminRoutes["/admin/*"]

    V1 --> ChatComp["POST /v1/chat/completions"]
    V1 --> Embeddings["POST /v1/embeddings"]
    V1 --> RAGRoutes["/v1/rag/*"]

    RAGRoutes --> Ingest["POST /v1/rag/ingest"]
    RAGRoutes --> ListDocs["GET /v1/rag/documents"]
    RAGRoutes --> DeleteDoc["DELETE /v1/rag/documents/{id}"]

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
        text language
        text category
    }

    chunks {
        text id PK
        text document_id FK
        int chunk_index
        text content
        text section_title
    }

    chunk_embeddings {
        text chunk_id FK
        blob embedding
        int dimension
    }

    backends {
        text id PK
        text name UK
        text base_url
        text api_key
        bool enabled
        bool is_llama_cpp
    }

    crawl_sources {
        text id PK
        text name
        text seed_url
        text category
        text language
        bool enabled
    }

    crawl_pages {
        text id PK
        text source_id FK
        text url
        text content_hash
        text doc_id FK
        text status
    }

    crawl_runs {
        text id PK
        text status
        int pages_discovered
        int pages_ingested
    }

    documents ||--o{ chunks : "has"
    chunks ||--|| chunk_embeddings : "has"
    crawl_sources ||--o{ crawl_pages : "has"
    crawl_pages }o--o| documents : "references"
```
