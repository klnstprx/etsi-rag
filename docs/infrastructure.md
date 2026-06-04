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

Object key schema:
- `crawl/{html|pdf}/{url_sha256}` — raw fetched bytes (re-processable)
- `crawl/img/{url_sha256}/{ref}.png` — rendered image artifacts (`ref` is `p1`, `p2`, … for PDF pages or `full` for HTML screenshots)

## System Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `poppler` | `pdftotext` (PDF text extraction), `pdfinfo` (PDF metadata), `pdftoppm` (PDF page → PNG renders) | `paru -S poppler` |
| `chromium` | Headless browser for JS-rendered pages and full-page screenshots (go-rod) | `paru -S chromium` |

## Build Dependencies

| Tool | Purpose |
|------|---------|
| Go 1.25+ | Compiler |
| `templ` | Template compiler (`.templ` -> `_templ.go`) |
| `tailwindcss` | CSS (standalone CLI, no Node.js) |
| `golangci-lint` | Linter (optional) |

## Containerized Deployment

A `Dockerfile` and `docker-compose.yml` (repo root) package the three runtime
services for third-party deployment:

| Service        | Image                                    | Role |
|----------------|------------------------------------------|------|
| `app`          | built from `Dockerfile`                  | the Go server + admin UI |
| `llama`        | `ghcr.io/ggml-org/llama.cpp:server-cuda` | chat / embedding / rerank backend |
| `rustfs`       | `rustfs/rustfs:latest`                   | S3-compatible object store |
| `createbucket` | `minio/mc:latest`                        | one-shot; creates the `etsi-rag` bucket |

Only the `app` image is built locally; the backend and object store reference
official images. llama-server and its model weights are **not** bundled — they
are hardware-specific and multi-GB, so the operator supplies them.

### Image design

- **Multi-stage build.** Stage 1 (`golang:1.25-bookworm`) installs the `templ`
  and `tailwindcss` CLIs, regenerates the embedded admin assets, and compiles a
  static `CGO_ENABLED=0` binary. Stage 2 (`debian:bookworm-slim`) carries only
  the binary plus the crawler's runtime deps.
- **Not scratch/alpine.** The crawler shells out to `poppler-utils`
  (`pdftoppm`/`pdfinfo`) and drives a system `chromium` via go-rod, so the
  runtime image installs both.
- **Runs unprivileged** (uid 10001). Chromium refuses to start as root without
  `--no-sandbox`; as a non-root user it works under Docker's default seccomp.
- **Config seeding.** `docker-entrypoint.sh` copies `config.default.toml` →
  `config.toml` on first boot (the binary's `config.Load` does not fall back to
  the template at runtime). docker-compose bind-mounts
  `deploy/config.docker.toml` over it, which differs from the default only in
  network topology: `server.host = 0.0.0.0` and service DNS names
  (`http://llama:1337/v1`, `http://rustfs:9000`).

### Quick start

```bash
# 1. Install the NVIDIA Container Toolkit (for the llama GPU service).
# 2. Drop .gguf weights into ./models — filenames must match the model = '...'
#    values in deploy/config.docker.toml (router resolves by filename).
docker compose up --build
```

Admin UI at `http://localhost:3000/admin`, RustFS console at
`http://localhost:9001`. Override any scalar config value via env, e.g.
`APP__SERVER__PORT=8080` (note: `[[backends]]` is an array — change backend
URLs in `deploy/config.docker.toml`, not via env).

### Caveats

- **GPU.** The compose file uses a service-level `devices: ["nvidia.com/gpu=all"]`
  CDI entry — the one GPU form portable across **both** Docker and Podman (the
  compose `devices` field accepts CDI device names on both runtimes; the older
  `deploy.resources` / `driver: cdi` block is Docker-only and fails under
  podman-compose). Host prerequisite: install the NVIDIA Container Toolkit and
  generate a CDI spec once with
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` (toolkit ≥ 1.18
  auto-refreshes it). Docker 28+ reads CDI by default; 25–27 needs
  `{"features": {"cdi": true}}` in `/etc/docker/daemon.json`. For CPU-only, drop
  the `devices` block and switch the image to
  `ghcr.io/ggml-org/llama.cpp:server-intel` (much slower). On SELinux-enforcing
  hosts, add `security_opt: ["label=disable"]` to the `llama` service.
- **`depends_on` conditions** (`service_completed_successfully`) have partial
  support under podman-compose; if the `createbucket` gate misbehaves there,
  start `rustfs` first or create the bucket manually via the console.
- **In-container crawling.** Headless Chromium in a container can need extra
  privileges (e.g. `--cap-add=SYS_ADMIN` or a relaxed seccomp profile) on some
  hosts. Query-only deployments over an already-populated corpus are unaffected.
- **Model presets.** Per-model llama-server args (`--mmproj` for vision, custom
  context size) require a router preset INI via `--models-preset`; see
  [llama-server.md](llama-server.md).
