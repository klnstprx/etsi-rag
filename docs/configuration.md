# Configuration
<!-- packages: internal/config -->
<!-- related: docs/architecture.md, config/config.default.toml -->

Centralized configuration via TOML files and environment variables.

## Loading Order

```
config/config.default.toml  ->  config/{RUN_MODE}.toml  ->  APP__* env vars
```

Set `RUN_MODE` environment variable to load an overlay (e.g., `RUN_MODE=test` loads `config/test.toml`). Environment variables use `APP__` prefix with `__` as section separator (e.g., `APP__AGENT__TEMPERATURE=0.5`).

On first run, if `config/config.toml` does not exist, `config.default.toml` is copied into place; from then on `config.toml` is the user-editable file (gitignored).

## Pre-provisioning (no UI access)

For headless installs (remote box, container, CI) where you can't reach the admin UI, you have three paths in increasing precedence — the later one always wins:

1. **Drop a complete `config/config.toml`** into the project before first launch. Include every section the deployment needs, including `[eval]` if the Python harness will run there. This is the simplest path and is what `ensureConfigFile` will pick up on first boot (no template copy).
2. **Use a `RUN_MODE` overlay**: ship the baseline as `config.toml` and put environment-specific deltas in `config/<mode>.toml`, then start the server with `RUN_MODE=<mode>`. Useful for keeping secrets and host-specific endpoints out of the baseline.
3. **`APP__*` env vars** for last-mile injection (secrets, port pinning, etc.). Example: `APP__AGENT__TEMPERATURE=0.3`, `APP__OBJECT_STORE__SECRET_KEY=...`.

Admin-UI saves only rewrite the sections the Go config struct owns; foreign top-level sections (notably `[eval]`) round-trip unchanged. So a pre-provisioned `[eval]` block survives any later edit performed through the UI. See [Persistence](#persistence).

## Sections at a glance

Sections are grouped by role:

| Role               | Sections                                                   |
| ------------------ | ---------------------------------------------------------- |
| **Infrastructure** | `[server]`, `[admin]`, `[database]`, `[object_store]`, `[[backends]]` |
| **Pipeline**       | `[chunking]`, `[embedding]`, `[retrieval]`, `[reranker]`, `[agent]`, `[agent.vision]`, `[session]` |
| **Ingest**         | `[crawler]`, `[crawler.pdf]`                               |
| **Eval (Python)**  | `[eval]` — consumed by `eval/` only; the Go server ignores it |

## TOML Reference

```toml
# ---------- Infrastructure ----------

[server]
host = '127.0.0.1'
port = 3000
read_timeout = 15             # seconds
write_timeout = 120           # seconds
idle_timeout = '60s'          # keep-alive idle timeout
shutdown_timeout = '30s'      # graceful shutdown grace period

[admin]
backend_probe_timeout = '3s'     # per-request timeout for backend health check
backend_models_timeout = '5s'    # per-request timeout for backend /models listing
chat_stream_timeout = '5m'       # max lifetime of an admin chat turn goroutine (orphaned-goroutine safety net)

[database]
path = 'data/etsi-rag.db'

[object_store]
endpoint = 'http://localhost:9000'
bucket = 'etsi-rag'
access_key = 'minioadmin'
secret_key = 'minioadmin'
region = 'us-east-1'

[[backends]]
name = 'local-llama'
base_url = 'http://localhost:1337/v1'
api_key = ''
enabled = true
is_llama_cpp = true           # enables llama.cpp /tokenize endpoint
supports_vision = false       # set true when the configured model has an mmproj/vision adapter loaded;
                              # admin UI shows a hint when /props detects --mmproj on the model

# ---------- Pipeline (chunk → embed → retrieve → answer) ----------

[chunking]
max_chunk_size = 256          # in tokens (not characters)
chunk_overlap = 32            # in tokens
min_chunk_size = 64           # minimum chunk size in tokens
separators = ["\n\n", "\n", '. ', ' ', '']  # fallback split separators for RecursiveTokenChunker

[embedding]
backend = 'local-llama'       # references backend by name
model = 'embedding-model'
dimension = 2560
max_tokens = 512              # embedding context window
batch_size = 2                # texts per embedding API call

[retrieval]
top_k = 10                   # number of chunks to retrieve
max_context_tokens = 2048    # token budget for retrieved context
                             # invariant: chunking.max_chunk_size must be <= max_context_tokens
                             # (validated at load time; otherwise a single chunk can starve
                             # the retrieval budget)
retrieval_mode = "dense"     # "dense"  — cosine similarity over query embeddings (default).
                             # "hybrid" — dense + BM25 keyword search merged via reciprocal-
                             #            rank fusion. See docs/decisions/008-hybrid-retrieval.md.
                             # Validated at load time; unknown values reject.

[reranker]                    # cross-encoder reranker (default-on; ADR-009)
enabled = true                # toggle on/off without restart (admin UI live-rebuilds)
backend = 'local-llama'       # MUST have is_llama_cpp = true; /rerank is a llama.cpp/Jina extension
model = 'bge-reranker-v2-m3'  # BAAI 568M multilingual cross-encoder; selected after eval sweep
top_n = 30                    # candidate pool size; must be >= retrieval.top_k
                              # (validated at load time and at admin save)
max_tokens = 0                # 0 = auto-detect via /props (n_ctx); used as the per-doc truncation budget
reserved_tokens = 16          # BOS/EOS/separator budget held back from per-doc truncation
timeout = '30s'               # per-request deadline; the default is cold-load forgiving
                              # because router-loaded reranker models load lazily

[agent]
backend = 'local-llama'
model = 'chat-model'
max_tokens = 1024             # output token limit
temperature = 0.7
context_window = 0            # total context window (auto-detected for llama.cpp, 0 = unknown)
max_iterations = 5            # max tool-calling iterations per turn
tool_output_max_tokens = 0    # per-tool-result token ceiling; 0 = auto (context_window/4, min 256)
passthrough = false           # true = accept client messages directly, no session management
system_prompt = ''            # custom system prompt text/template (empty = use embedded prompts/agent.tmpl)
                              # available template vars: {{.DocumentCount}}, {{.ChunkCount}}

[agent.vision]
max_images_per_call = 6           # cap on image artifacts attached per tool call (vision-capable backends only)
max_image_bytes_per_call = 65536  # cap on total image_url URL bytes attached per tool call
public_base_url = ''              # absolute base URL prefixed onto /v1/artifacts/{token}.png so the LLM
                                  # backend can fetch image content blocks (e.g. 'http://localhost:8080');
                                  # empty disables image attachment regardless of backend.supports_vision

[session]
max_messages = 10             # max messages per session
ttl_seconds = 3600            # session expiry (1 hour)

# ---------- Ingest ----------

[crawler]
request_delay = '500ms'
request_timeout = '30s'
progress_log_interval = '30s' # how often the running crawl logs a progress line
max_depth = 10
max_pages = 5000
concurrency = 4
max_body_size = 10485760      # 10 MB
min_content_length = 200      # skip pages shorter than this (chars), 0 = disabled
embed_workers = 1             # concurrent embed stage workers
parse_workers = 4             # concurrent parse stage workers
max_embed_fails = 3           # consecutive embed failures before aborting crawl
js_patterns = ['/centers/subject/']
rejection_patterns = [        # case-insensitive substrings that mark a page as low-quality
    'página no encontrada',                 # soft 404
    'error 404',
    'página en construcción',
    'sitio en construcción',
    'sin descripción disponible',           # CMS placeholder
]
strip_query_params = [        # case-insensitive query-param names stripped during URL normalization
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term',
    'fbclid', 'gclid', '_ga', 'ref', 'lang',
]

[crawler.pdf]
heading_patterns = [          # substring patterns for known PDF section headers
    'DESCRIPCION', 'OBJETIVOS', 'CONTENIDOS', 'METODOLOGIA',
    'EVALUACION', 'COMPETENCIAS', 'TEMARIO', 'BIBLIOGRAFIA',
    'ACTIVIDADES FORMATIVAS', 'SISTEMA DE EVALUACION',
    'RESULTADOS DE APRENDIZAJE', 'EQUIPO DOCENTE',
    'RECOMENDACIONES', 'DISTRIBUCION DEL TRABAJO',
]
render_dpi = 120              # pdftoppm -r DPI for PDF page-image renders; 0 disables image rendering

# ---------- Eval ----------
# [eval] is consumed only by the eval/ harness (Python). The Go server ignores it.
# The harness refuses to start if [eval] is missing or malformed.
# judge_backend must reference an enabled [[backends]] entry; pick a different
# backend/model from [agent] to avoid self-preference bias when the same model
# grades its own answers.
[eval]
judge_backend = 'remote-claude'
judge_model   = 'claude-sonnet-4-6'
```

## Backend Resolution

`server.Server.resolveEmbedClient()` and `resolveChatClient()` (in `internal/server/wiring.go`) are called both at startup and on every admin-UI config save (via `rebuildClients()`):
1. Look up the backend by name from the `[[backends]]` array
2. Merge stage-specific config (model, batch_size for embedding; model, max_tokens, temperature for agent)
3. Build an `llm.EmbedClient` or `llm.ChatClient` with the resolved settings

`resolveCounter()` creates the appropriate token counter:
- `is_llama_cpp = true` -> `LlamaCppCounter` (HTTP `/tokenize` endpoint)
- `is_llama_cpp = false` -> `TiktokenCounter` (tiktoken-go library)
- Fallback -> `EstimateCounter` (len/4)

## Auto-Detection (llama.cpp)

`probeLlamaCppBackends()` probes each unique llama.cpp backend once at startup and after config changes:
- **Embedding**: auto-detects `max_tokens` (n_ctx) and `dimension` (n_embd)
- **Agent**: auto-detects `context_window` (n_ctx)
- **Reranker**: auto-detects `max_tokens` (n_ctx); also has the side-effect of asking the router to load the reranker model so the first `/rerank` request avoids cold-load latency
- Logs a warning if `max_tokens` exceeds the probed `context_window`
- Non-llama.cpp backends are skipped; set `context_window` manually if needed

## Persistence

`config.WriteTo()` serializes the current config back to TOML. Used by the admin UI's config editor to persist changes.

It treats `config.toml` as a shared file: it loads what's on disk, drops the top-level keys that `Config` owns (so a removed `[[backends]]` entry actually disappears instead of surviving koanf's path-merge), then overlays the typed struct. Top-level sections not declared on `Config` — `[eval]` and anything else a sidecar tool may add — pass through untouched.

## Legacy keys (post-restructure)

The TOML layout was reorganized to group infrastructure / pipeline / ingest sections. `Load()` rejects pre-restructure keys with a rename hint instead of silently zero-defaulting them. If startup fails with `config uses legacy keys`, rename:

| Old                                  | New                                    |
| ------------------------------------ | -------------------------------------- |
| `[rag] top_k`                        | `[retrieval] top_k`                    |
| `[rag] max_context_tokens`           | `[retrieval] max_context_tokens`       |
| `[rag] retrieval_mode`               | `[retrieval] retrieval_mode`           |
| `[api] passthrough`                  | `[agent] passthrough`                  |
| `[agent] public_base_url`            | `[agent.vision] public_base_url`       |
| `[agent] max_images_per_call`        | `[agent.vision] max_images_per_call`   |
| `[agent] max_image_bytes_per_call`   | `[agent.vision] max_image_bytes_per_call` |
