# etsi-eval

LLM-as-judge evaluation harness for the etsi-rag agent.

## How it works

For each question in `dataset.jsonl`:

1. Look up the reference document(s) in `data/etsi-rag.db` by each entry in `golden_urls` (read-only). URLs are canonicalised on both sides (see "URL canonicalisation" below) so cosmetic differences don't cause silent misses. Skipped for `expected_behavior == "refuse"` rows.
2. Probe the retrieval layer via `POST /v1/rag/query` to compute Recall@k / MRR against `golden_urls` (URL-level, k = 5 and 10). Skipped for refuse rows where the metric is undefined.
3. Ask the agent via `POST /v1/chat/completions` — single user turn. Each row gets a unique `X-Session-ID` of the form `eval-<run_id>-<question_id>` so chat history never bleeds between questions even when the server is in session mode.
4. Feed `(question, agent answer, reference docs, expected_behavior)` to the **judge LLM** (configured separately in `[eval]` — see "Configuring the judge" below; called via the OpenAI Python SDK so the judge never goes through the agent's tool loop). Transient failures are handled by the SDK's built-in retry layer.
5. The judge returns three independent sub-scores (see "Rubric" below) constrained by an OpenAI structured-output JSON schema, so the response is always parseable. For refuse rows, the rubric scores how well the model declined to answer.
6. Results are aggregated per dimension, retrieval metric, category, and difficulty and written to `results/TIMESTAMP.json` (schema_version 2).

## Rubric

The judge scores each answer on three independent dimensions:

| Dimension | Question it answers |
|---|---|
| **faithfulness** | Is every claim in the answer entailed by the reference document? Catches hallucination. |
| **correctness**  | Of the claims that are made, are they factually accurate? Catches subtle wrongness. |
| **completeness** | Does the answer address every part of the question with the key facts present? Catches partial answers. |

`overall = min(faithfulness, correctness, completeness)`. Min (not mean) is the policy: an unfaithful answer is worthless even if complete; the weakest dimension caps quality. All four numbers are reported.

For `expected_behavior == "refuse"` rows the rubric is repurposed: faithfulness penalises fabricated content, correctness rewards explicit acknowledgement of lack of information, completeness rewards clear communication of the limitation.

## Retrieval metrics

Computed against the `golden_urls` field on each row (URL-level, not chunk-id, so they survive re-crawls):

| Metric | Definition |
|---|---|
| **Recall@5**, **Recall@10** | Fraction of golden URLs that appear in the first k retrieved URLs. For multi-URL goldens, this is `\|retrieved[:k] ∩ golden\| / \|golden\|`. |
| **MRR** | Reciprocal rank of the earliest retrieved URL that is a golden URL. |

The retrieval probe (`POST /v1/rag/query`) runs separately from the agent and reflects the raw retrieval layer, not what the agent's reformulated tool query produced. This separation is intentional: it lets us ablate the retriever (chunking, embedding model, top_k) independently of the agent's tool-use behaviour. Refuse rows are excluded from retrieval aggregates (golden_urls is empty → metric undefined).

**Single- vs multi-golden caveat:** The vast majority of rows have a single `golden_urls` entry, so for those rows Recall@k is binary (0 or 1) and the aggregate mean is effectively a hit-rate. True recall (a value strictly between 0 and 1) is only meaningful on the cross-doc subset, where two or more URLs need to surface together. Numbers should be interpreted accordingly in any write-up.

## URL canonicalisation

Ground-truth matching applies a small set of conservative normalisations on both the dataset's `golden_urls` and the URLs returned by `/v1/rag/query`, so cosmetic differences (trailing slash, scheme/host case, default ports `:80`/`:443`, fragments) never cause a silent miss. Query strings are left untouched. The DB lookup tries an exact match first and falls back to a one-shot canonical-form scan over the `documents` table, cached per process.

Run with `--validate-urls` to fail fast before any agent calls if a `golden_urls` entry can't be resolved against the indexed corpus — useful after a fresh crawl.

## Configuring the judge

A `[eval]` section is **required** in `config/config.toml`. The judge_backend must reference an enabled `[[backends]]` entry. The eval tool will refuse to start if the section is missing or malformed. The backend must be OpenAI-compatible — the judge is invoked via the OpenAI Python SDK with `response_format=json_schema` so the verdict shape is enforced at decode time. llama-server, Anthropic's OpenAI-compat endpoint, and OpenAI itself all qualify.

```toml
[[backends]]
name = "remote-claude"
base_url = "https://api.anthropic.com/v1"
api_key = "sk-..."
enabled = true

[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"

# Optional — USD per million tokens. When set, the runner prints a running
# cost estimate per row and a final total. Both default to 0 (disabled).
judge_input_price_per_mtok  = 3.0
judge_output_price_per_mtok = 15.0
```

Using the same backend/model for both the agent and the judge is allowed but discouraged — it's the canonical setup for self-preference bias and the runner will print a warning. For thesis-grade results, the judge should be a different model family from the agent (e.g. agent = local Gemma, judge = remote Claude).

## Token usage and cost

Every judge call returns an OpenAI-compatible `usage` block (`prompt_tokens`, `completion_tokens`, `total_tokens`). The runner:

- stores each row's tokens on `Row.judge_prompt_tokens` / `judge_completion_tokens` / `judge_total_tokens`;
- shows per-row tokens in the live status line (`toks=1234+45`) plus a cumulative count (`cum=8923`);
- aggregates totals into `aggregate.judge_usage` in the output JSON;
- if `judge_input_price_per_mtok` / `judge_output_price_per_mtok` are set in `[eval]`, also displays per-row running cost (`cost=$0.0234`) and a final estimate.

If a backend strips the `usage` block (rare, but happens behind some proxies), the runner falls back to zero — totals will under-report rather than crash.

## Prerequisites

- Python 3.11+ (uses `tomllib` from stdlib).
- `uv` on PATH.
- A running etsi-rag server (default `http://127.0.0.1:3000`).
- A populated `data/etsi-rag.db` (crawl has been run at least once).
- The backend referenced by `[agent].backend` in `config/config.toml` must be up.

## Archived runs

`results/` keeps a few human-named runs as reference points:

| File | Config | Overall | Notes |
|---|---|---|---|
| `baseline_topk10_ctx4092.json` | `[retrieval] top_k=10, max_context_tokens=4092, retrieval_mode="dense"` | 76.1 | Reference baseline (2026-05-04). |
| `regression_topk20_ctx8192.json` | `[retrieval] top_k=20, max_context_tokens=8192, retrieval_mode="dense"` | 45.5 | Negative control: wider context regresses gemma4-e4b by 31 points. See `docs/decisions/007-rag-context-budget.md`. |
| `hybrid_topk10_ctx4092.json` | `[retrieval] top_k=10, max_context_tokens=4092, retrieval_mode="hybrid"` | **80.2** | BM25 + dense via RRF. +4.1 pts Overall, +12.8 on `general` category. See `docs/decisions/008-hybrid-retrieval.md`. |

Per-run timestamped files (`YYYYMMDD_HHMMSS.json`) are working artifacts; rename to a descriptive name once a run is worth keeping.

## Run

```fish
cd etsi-rag/eval
uv sync

# full run over the entire dataset
uv run python -m etsi_eval.runner

# filters
uv run python -m etsi_eval.runner --difficulty easy
uv run python -m etsi_eval.runner --category course
uv run python -m etsi_eval.runner --ids 1,7,12

# retrieval probe top_k (default 10)
uv run python -m etsi_eval.runner --retrieval-top-k 20

# pre-flight: fail fast if any golden_urls entry isn't in the DB
uv run python -m etsi_eval.runner --validate-urls

# parallel execution (requires llama-server -np N on the agent side)
uv run python -m etsi_eval.runner --concurrency 4

# explicit output path
uv run python -m etsi_eval.runner --out /tmp/run1.json
```

## Parallel execution

By default the runner is single-flight (`--concurrency 1`) — one row at a time. With `--concurrency N` it runs N rows in parallel via a thread pool, which is useful because each row is dominated by network I/O on the agent (local llama-server) and the judge (remote LLM).

Important caveat: a local llama-server only delivers parallelism when started with `--parallel N` (server slots) — otherwise extra concurrent requests just queue inside the server and you get no speedup. Continuous batching is enabled by default in llama-server, so once the slots are there the GPU schedules requests automatically. Set `N` to match (or exceed) the eval `--concurrency`. The remote judge backends used in this repo (Anthropic) handle concurrent requests natively without extra setup.

Row order in the output JSON is always sorted by `id`, regardless of completion order.

## Output

Console: progress line per question, then a summary table with per-category / per-difficulty means.

JSON file (default `results/YYYYMMDD_HHMMSS.json`):

```json
{
  "schema_version": 2,
  "timestamp": "...",
  "run_id": "a1b2c3d4",
  "server_base_url": "http://127.0.0.1:3000",
  "agent_backend": "local-llama",
  "agent_model": "agent-model",
  "judge_backend": "remote-claude",
  "judge_model": "claude-sonnet-4-6",
  "aggregate": {
    "total": 85,
    "scored": 83,
    "errored": 2,
    "by_dimension": {
      "faithfulness": {"count": 83, "mean": 82.1, "min": 30, "max": 100},
      "correctness":  {"count": 83, "mean": 78.4, "min": 20, "max": 100},
      "completeness": {"count": 83, "mean": 70.5, "min": 0,  "max": 100},
      "overall":      {"count": 83, "mean": 65.2, "min": 0,  "max": 95}
    },
    "retrieval": {
      "recall_at_5":  {"count": 75, "mean": 0.722, "min": 0.0, "max": 1.0},
      "recall_at_10": {"count": 75, "mean": 0.833, "min": 0.0, "max": 1.0},
      "mrr":          {"count": 75, "mean": 0.611, "min": 0.0, "max": 1.0}
    },
    "by_category":   { "course": {...}, ... },
    "by_difficulty": { "easy": {...}, ... }
  },
  "rows": [
    {
      "id": 1,
      "faithfulness": 95, "correctness": 90, "completeness": 80, "overall": 80,
      "faithfulness_rationale": "...",
      "agent_answer": "...",
      "errors": [],
      ...
    }
  ]
}
```

The `errors` field on each row is a list (not a single string) so a row that hits both a transient retrieval failure and a judge failure retains both messages, each prefixed with its stage (`"retrieval: ..."`, `"agent: ..."`, `"judge: ..."`).

## Dataset format

One JSON object per line:

```json
{
  "id": 1,
  "category": "course",
  "difficulty": "easy",
  "question": "¿Quién coordina ...?",
  "key_facts": ["Rodrigo Roman Castro", "rroman@uma.es"],
  "expected_behavior": "answer",
  "golden_urls": ["https://sara.uma.es/..."],
  "notes": ""
}
```

Fields:
- `golden_urls` — list of canonical URLs that should be used as ground truth. For cross-doc questions, list all URLs whose chunks should appear in retrieval (eval Step 4 will compute Recall@k against this list). Must be empty when `expected_behavior == "refuse"`.
- `expected_behavior` — `"answer"` (default) when the corpus contains the answer; `"refuse"` when the question is deliberately out-of-corpus and the system should acknowledge it doesn't know. The judge applies a different rubric for `"refuse"` rows.
- `key_facts` — retained for human review and curation; **not** passed to the judge.
- `notes` — curator-only; ignored by the runner and judge.

Legacy rows that use `source_url` (single string) are auto-promoted to `golden_urls = [that_url]` at load time, but new rows should use the canonical shape above.

## Extending

- **More questions**: append to `dataset.jsonl`. IDs must be unique.
- **Different judge model**: edit `[eval].judge_backend` / `[eval].judge_model` in `config/config.toml`.
- **Different grading rubric**: edit `JUDGE_SYSTEM` in `etsi_eval/judge.py`.
- **Retrieval-layer introspection** (what the agent actually retrieved vs. the ground-truth source): add a second call to `/v1/rag/query` and compare chunks against `source_url`.

## Tests

```fish
uv run pytest
```

Just the parsing/dataset smoke tests — no network.
