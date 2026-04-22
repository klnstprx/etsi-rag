# Configuration
<!-- packages: internal/config -->
<!-- related: docs/architecture.md, config/config.default.toml -->

Centralized configuration via TOML files and environment variables.

## Loading Order

```
config/config.default.toml  ->  config/{RUN_MODE}.toml  ->  APP__* env vars
```

Set `RUN_MODE` environment variable to load an overlay (e.g., `RUN_MODE=test` loads `config/test.toml`). Environment variables use `APP__` prefix with `__` as section separator (e.g., `APP__AGENT__TEMPERATURE=0.5`).

## TOML Reference

```toml
[server]
host = '127.0.0.1'
port = 3000
read_timeout = 15             # seconds
write_timeout = 120           # seconds

[[backends]]
name = 'local-llama'
base_url = 'http://localhost:1337/v1'
api_key = ''
enabled = true
is_llama_cpp = true           # enables llama.cpp /tokenize endpoint

[embedding]
backend = 'local-llama'       # references backend by name
model = 'embedding-model'
dimension = 4096
max_tokens = 512              # embedding context window
batch_size = 2                # texts per embedding API call

[agent]
backend = 'local-llama'
model = 'chat-model'
max_tokens = 1024             # output token limit
temperature = 0.7
context_window = 0            # total context window (auto-detected for llama.cpp, 0 = unknown)
max_iterations = 5            # max tool-calling iterations per turn
system_prompt = ''            # custom system prompt text/template (empty = use embedded prompts/agent.tmpl)
                              # available template vars: {{.DocumentCount}}, {{.ChunkCount}}

[database]
path = 'data/etsi-rag.db'

[rag]
top_k = 5                    # number of chunks to retrieve
max_context_tokens = 2048    # token budget for retrieved context

[api]
passthrough = false           # true = accept client messages directly, no session management

[chunking]
max_chunk_size = 256          # in tokens (not characters)
chunk_overlap = 32            # in tokens
min_chunk_size = 64           # minimum chunk size in tokens
separators = ["\n\n", "\n", '. ', ' ', '']  # fallback split separators for RecursiveTokenChunker

[session]
max_messages = 10             # max messages per session
ttl_seconds = 3600            # session expiry (1 hour)

[crawler]
request_delay = '500ms'
request_timeout = '30s'
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

[object_store]
endpoint = 'http://localhost:9000'
bucket = 'etsi-rag'
access_key = 'minioadmin'
secret_key = 'minioadmin'
region = 'us-east-1'
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
- Logs a warning if `max_tokens` exceeds the probed `context_window`
- Non-llama.cpp backends are skipped; set `context_window` manually if needed

## Persistence

`config.WriteTo()` serializes the current config back to TOML. Used by the admin UI's config editor to persist changes.
