# Configuration
<!-- packages: internal/config -->
<!-- related: docs/architecture.md, config/config.default.toml -->

Centralized configuration via TOML files and environment variables.

## Loading Order

```
config/config.default.toml  ->  config/{RUN_MODE}.toml  ->  APP__* env vars
```

Set `RUN_MODE` environment variable to load an overlay (e.g., `RUN_MODE=test` loads `config/test.toml`). Environment variables use `APP__` prefix with `__` as section separator (e.g., `APP__CHAT__TEMPERATURE=0.5`).

## TOML Reference

```toml
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

[chat]
backend = 'local-llama'
model = 'chat-model'
max_tokens = 1024
temperature = 0.7

[agent]
backend = 'local-llama'
model = 'chat-model'
max_tokens = 1024
temperature = 0.7

[chunking]
chunk_size = 256              # in tokens (not characters)
chunk_overlap = 32            # in tokens
min_chunk_size = 64           # minimum chunk size in tokens

[crawler]
request_delay = '500ms'
request_timeout = '30s'
max_depth = 10
max_pages = 2000
concurrency = 4
max_body_size = 10485760      # 10 MB
js_patterns = ['/centers/subject/']

[object_store]
endpoint = 'http://localhost:9000'
bucket = 'etsi-rag'
access_key = 'minioadmin'
secret_key = 'minioadmin'
region = 'us-east-1'
```

## Backend Resolution

`main.go` uses `clientForStage()` to:
1. Look up the backend by name from `[[backends]]` array
2. Merge stage-specific config (model, max_tokens, temperature)
3. Build an `llm.Client` with the resolved settings

`counterForBackend()` creates the appropriate token counter:
- `is_llama_cpp = true` -> `LlamaCppCounter` (HTTP `/tokenize` endpoint)
- `is_llama_cpp = false` -> `TiktokenCounter` (tiktoken-go library)
- Fallback -> `EstimateCounter` (len/4)

## Persistence

`config.WriteTo()` serializes the current config back to TOML. Used by the admin UI's config editor to persist changes.
