# etsi-rag

RAG chatbot for ETSI Informatica (Universidad de Malaga). Final degree project (TFG).

Crawls the UMA website, extracts and chunks content, embeds it into a vector store, and provides an AI assistant that answers student questions with retrieved context.

## Quick Start

```bash
# Dependencies
paru -S poppler chromium

# Build
make build

# Configure
cp config/config.default.toml config/config.local.toml
# Edit config/config.local.toml with your backend settings

# Run
make run
```

Requires an OpenAI-compatible LLM server (e.g., [llama-server](https://github.com/ggml-org/llama.cpp)) and [RustFS](https://github.com/nicholaskell/rustfs) for object storage.

## Development

```bash
make dev           # Watch mode (templ + tailwind + go run)
make test          # Run all tests
make lint          # golangci-lint
make clean         # Reset DB + build artifacts
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Architecture](docs/architecture.md) | System design, package overview, data flow |
| [Configuration](docs/configuration.md) | TOML reference, env vars, backend resolution |
| [Crawler](docs/crawler.md) | BFS crawler design, pipeline, content extraction |
| [UMA Site Structure](docs/uma-site-structure.md) | URL patterns, seed URLs, content types |
| [Go Guidelines](docs/go-guidelines.md) | Go Proverbs + project conventions |
| [Infrastructure](docs/infrastructure.md) | llama-server, RustFS, system deps |
| [llama-server Reference](docs/llama-server.md) | Full llama-server documentation |

### Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [001](docs/decisions/001-no-cgo.md) | No CGO — pure Go SQLite (`modernc.org/sqlite`) |
| [002](docs/decisions/002-brute-force-cosine.md) | Brute-force cosine similarity (no ANN index) |
| [003](docs/decisions/003-token-aware-chunking.md) | Token-aware, structure-aware chunking |
| [004](docs/decisions/004-markdown-pipeline.md) | Markdown throughout the pipeline |
| [005](docs/decisions/005-split-llm-clients.md) | Split LLM clients per stage |

## Stack

- **Go** — backend, no CGO
- **SQLite** — documents, chunks, embeddings (modernc.org/sqlite)
- **templ + Tailwind + HTMX** — admin UI
- **openai-go** — LLM SDK
- **go-rod** — headless Chrome for JS pages
- **go-readability** — content extraction
