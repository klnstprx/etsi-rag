# Model weights

Drop your `.gguf` model files here under the filenames below. The `llama`
compose service mounts this directory read-only at `/models` and runs
llama-server in router mode with [`../deploy/presets.ini`](../deploy/presets.ini),
which maps each file to the model ID that
[`../deploy/config.docker.toml`](../deploy/config.docker.toml) uses and sets
its context, slots and GPU placement. Models load on first use.

| Stage     | Model ID             | File                                  |
| --------- | -------------------- | ------------------------------------- |
| agent     | `gemma4-12b`         | `gemma-4-12b-it-UD-Q8_K_XL.gguf`      |
| agent     | (vision projector)   | `gemma-4-12b-it-mmproj-F32.gguf`      |
| embedding | `octen-embedding-4b` | `Octen-Embedding-4B.Q8_0.gguf`        |
| reranker  | `bge-reranker-v2-m3` | `bge-reranker-v2-m3-Q8_0.gguf`        |

The projector ships in the Gemma repository as `mmproj-F32.gguf`; rename it
on download so it does not clash with other models' projectors. To use other
weights, edit the paths in `deploy/presets.ini`.

Weights are gitignored; this directory is tracked only via `.gitkeep`.
