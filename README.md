# etsi-rag

RAG chatbot for ETSI Informatica (Universidad de Malaga). Final degree project (TFG).

Crawls the UMA website, extracts and chunks content, embeds it into a vector store, and provides an AI assistant that answers student questions with retrieved context.

## Quick Start

```bash
# Runtime deps (poppler for PDF rendering, chromium for JS-rendered pages)
sudo apt install poppler-utils chromium    # Debian/Ubuntu
brew install poppler chromium              # macOS
paru -S poppler chromium                   # Arch

# Build tooling: `make build` runs templ at the version pinned in go.mod
# (via `go run`), so only tailwindcss must be on $PATH.
# tailwindcss: download the standalone CLI for your platform onto $PATH
# https://github.com/tailwindlabs/tailwindcss/releases

# Build
make build

# Run (creates config/config.toml from config.default.toml on first start)
make run

```

Requires an OpenAI-compatible LLM server (e.g., [llama-server](https://github.com/ggml-org/llama.cpp)) and [RustFS](https://github.com/nicholaskell/rustfs) for object storage.

## Configuration

The server reads `config/config.toml` (created from `config.default.toml` on first run). You can edit it by hand, but the easiest way is the **admin UI at `http://localhost:3000/admin`**, which exposes a config editor and lets you manage backends, run crawls, and chat with the agent. Most scalar config values are live-reloaded — no restart needed.

### Backends

LLM endpoints are defined as `[[backends]]` entries (one per OpenAI-compatible server, referenced by `name`). Each pipeline stage then points at a backend:

- `[embedding]` — **required**, must reference a backend + embedding model.
- `[agent]` — **required**, must reference a backend + a chat/instruct model (never an embedding model).
- `[reranker]` — optional (default-on), references a llama.cpp backend serving a cross-encoder rerank model.

A backend must exist and be `enabled` before a stage can use it; the admin UI surfaces health/probe status for each.

### Crawling

The store starts empty — content is ingested by crawling the configured sources. BFS seeds are defined as `[[sources]]` entries in `config/sources.toml` (created from `sources.default.toml` on first run); each has a stable `key`, a `seed_url`, and agent-facing `description`/`tags` hints. Edit the file directly or manage sources and launch crawls from the admin UI at `http://localhost:3000/admin/crawler`. See [docs/crawler.md](docs/crawler.md) for the pipeline details.

### LLM server (llama-server)

The agent, embedding, and reranker stages all talk to an OpenAI-compatible endpoint. In router mode, llama-server serves every model from one port, loading each on first reference:

```bash
llama-server --models-preset ~/Models/preset.ini --port 1337 --host 127.0.0.1
```

The preset assigns per-model settings (multimodal projector, context size, sampling, GPU placement) that don't belong in `config.toml`. Model paths must be absolute. The section names are the model identifiers the server exposes; the ones below match the `model` values in `config.default.toml`. This preset fits the whole stack on one 24 GB GPU (about 21.2 GiB):

```ini
version = 1

[gemma4-12b]
model = /home/you/Models/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q8_K_XL.gguf
mmproj = /home/you/Models/gemma-4-12b-it-GGUF/mmproj-F32.gguf
c = 32768
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 64
jinja = true
flash-attn = on
reasoning = off
n-gpu-layers = all
fit = off

[octen-embedding-4b]
model = /home/you/Models/Octen-Embedding-4B/Octen-Embedding-4B.Q8_0.gguf
embedding = true
c = 8192
parallel = 1
jinja = true
n-gpu-layers = all
fit = off

[bge-reranker-v2-m3]
model = /home/you/Models/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf
reranking = true
c = 8192
parallel = 1
b = 8192
ub = 8192
jinja = true
n-gpu-layers = all
fit = off
```

- **Context and slots:** without `c` and `parallel`, llama-server sizes each model's KV cache for its full training context times four slots. Octen alone then takes about 10 GB and the 12B no longer fits.
- **Placement:** `n-gpu-layers = all` with `fit = off` keeps every model fully on the GPU. Don't pass `-ngl` on the command line; it overrides `fit` for every model.
- **Reranker batch:** the reranker scores each query–passage pair in one physical batch, so its `b`/`ub` must cover its context; at llama-server's default of 512, longer pairs fail with HTTP 500. The embedder's inputs are single chunks (`[chunking] max_chunk_size`), well within the default.

### RustFS (object storage)

RustFS holds document artifacts (PDF page renders, HTML screenshots) for multimodal retrieval. Access is configured in the `[object_store]` section of `config.toml`; the defaults match the container below:

```toml
[object_store]
endpoint   = 'http://localhost:9000'
bucket     = 'etsi-rag'
access_key = 'minioadmin'
secret_key = 'minioadmin'
region     = 'us-east-1'
```

```bash
podman run -d \
  --name rustfs \
  -p 9000:9000 \
  -p 9001:9001 \
  -v rustfs-data:/data \
  -e RUSTFS_ACCESS_KEY=minioadmin \
  -e RUSTFS_SECRET_KEY=minioadmin \
  -e RUSTFS_CONSOLE_ENABLE=true \
  rustfs/rustfs:latest \
  --console-enable \
  /data

# Start existing container
podman start rustfs
```

The app does **not** create the bucket automatically — create the `etsi-rag` bucket (matching `[object_store].bucket`) once via the RustFS console at `http://localhost:9001` before ingesting content.

## Development

```bash
make dev           # Watch mode (templ + tailwind + go run)
make test          # Run all tests
make lint          # golangci-lint
make clean         # Remove build artifacts (does not touch the DB)
make reset-db      # Rename DB to a timestamped .bak so a fresh schema migrates
```

## Documentation

| Doc                                              | Description                                      |
| ------------------------------------------------ | ------------------------------------------------ |
| [Architecture](docs/architecture.md)             | System design, package overview, data flow       |
| [Configuration](docs/configuration.md)           | TOML reference, env vars, backend resolution     |
| [Retrieval](docs/retrieval.md)                   | Dense / BM25 / RRF formulas, parameters, sources |
| [Crawler](docs/crawler.md)                       | BFS crawler design, pipeline, content extraction |
| [UMA Site Structure](docs/uma-site-structure.md) | URL patterns, seed URLs, content types           |
| [Go Guidelines](docs/go-guidelines.md)           | Go Proverbs + project conventions                |
| [Infrastructure](docs/infrastructure.md)         | llama-server, RustFS, system deps                |
| [Eval Methodology](docs/eval-methodology.md)     | LLM-as-judge limits, defensibility paths         |
| [llama-server Reference](docs/llama-server.md)   | Full llama-server documentation                  |
| [Decisions](docs/decisions/)                     | Architecture Decision Records (ADRs)             |

## Stack

- **Go** — backend, no CGO
- **SQLite** — documents, chunks, embeddings (modernc.org/sqlite)
- **templ + Tailwind + HTMX** — admin UI
- **openai-go** — LLM SDK
- **chromedp** — headless Chrome for JS pages
- **go-readability** — content extraction
