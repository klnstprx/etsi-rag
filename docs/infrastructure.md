# Infrastructure
<!-- related: docs/configuration.md -->

External services and system dependencies required to run the project.

## LLM Backend (llama-server)

Any OpenAI-compatible server works. The project is designed around `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp).

- Provides `/v1/chat/completions`, `/v1/embeddings`, and `/tokenize` endpoints
- Set `is_llama_cpp = true` in backend config to enable the `/tokenize` endpoint for precise token counting
- Supports F16 and quantized models on GPU and CPU
- Features: parallel decoding, continuous batching, speculative decoding, function calling

See [docs/llama-server.md](llama-server.md) for the full llama-server reference.

### Typical Setup

```bash
# Chat model
llama-server -m model.gguf --port 1337

# Or with separate embedding model
llama-server -m embed-model.gguf --port 1338 --embeddings
```

## Object Storage (RustFS)

S3-compatible object storage for raw content archival. Using [RustFS](https://github.com/nicholaskell/rustfs) (MinIO-compatible).

```bash
# Default config
endpoint = 'http://localhost:9000'
bucket = 'etsi-rag'
access_key = 'minioadmin'
secret_key = 'minioadmin'
```

Object key schema: `crawl/{html|pdf}/{url_sha256}`

## System Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `poppler` | `pdftotext` for PDF text extraction | `paru -S poppler` |
| `chromium` | Headless browser for JS-rendered pages (go-rod) | `paru -S chromium` |

## Build Dependencies

| Tool | Purpose |
|------|---------|
| Go 1.25+ | Compiler |
| `templ` | Template compiler (`.templ` -> `_templ.go`) |
| `tailwindcss` | CSS (standalone CLI, no Node.js) |
| `golangci-lint` | Linter (optional) |
