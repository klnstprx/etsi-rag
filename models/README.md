# Model weights

Drop your `.gguf` model files here. The `llama` compose service mounts this
directory read-only at `/models` and runs llama-server in router mode
(`--models-dir /models`), which loads a model on first reference by filename.

Filenames **must match** the `model = '...'` values in
[`../deploy/config.docker.toml`](../deploy/config.docker.toml):

| Stage     | Default filename                  |
| --------- | --------------------------------- |
| embedding | `Octen-Embedding-4B.Q8_0.gguf`    |
| agent     | `gemma-4-12b-it-Q8_K_XL.gguf`     |
| reranker  | `bge-reranker-v2-m3` (gguf)       |

Per-model launch args (e.g. `--mmproj` for a vision model, custom `-c`
context size) require a router preset INI — see `docs/llama-server.md` and
the `--models-preset` flag, then mount it alongside the weights.

Weights are gitignored; this directory is tracked only via `.gitkeep`.
