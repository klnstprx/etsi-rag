# etsi-eval

Evaluation harness for the etsi-rag agent. Captures per-row raw evidence
(answer, tool calls, retrieval metrics); pairwise comparison between two
runs is the only judge-graded surface — see [Pairwise preference](#pairwise-preference) below.

> **Per-row LLM-as-judge scoring was retired 2026-05-06.** Absolute scores
> at this dataset size sat inside positional-flip noise (39–48 % unstable;
> see `docs/eval-methodology.md` §2). The runner now records evidence only;
> direction-of-change is decided by pairwise. Schema version 3.

## How it works

For each question in `dataset.jsonl`:

1. Look up the reference document(s) in `data/etsi-rag.db` by each entry in `golden_urls` (read-only). Used as a sanity-check that the eval is pointed at the indexed corpus we expect; not consumed by any judge in this runner. URLs are canonicalised on both sides (see "URL canonicalisation" below). Skipped for `expected_behavior == "refuse"` rows.
2. Probe the retrieval layer via `POST /v1/rag/query` to compute Recall@k / MRR against `golden_urls` (URL-level, k = 5 and 10). Skipped for refuse rows where the metric is undefined.
3. Ask the agent via `POST /v1/agent/chat` (the proprietary endpoint) — single user turn (`{"message": ...}`). Each row gets a unique `X-Session-ID` of the form `eval-<run_id>-<question_id>` so chat history never bleeds between questions. (We use the proprietary endpoint because eval needs the per-call tool audit trail. The OpenAI-compatible `/v1/chat/completions` keeps the standard schema — including `choices[].message.tool_calls`, which is always `null` here since the agent resolves tools server-side and returns only final text — but carries no proprietary root-level `tool_calls` audit field.)
4. Record the agent answer and the list of tool invocations the agent made (`tool_calls` in the proprietary response — `name`, `args`, `error`, `duration_ms` per call).
5. Aggregate retrieval metrics and per-tool counts. Results are written to `results/TIMESTAMP.json` (schema_version 3).

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

## Configuring the judge (for pairwise)

A `[eval]` section is **required** in `config/config.toml` so the pairwise runner has a judge to call. The runner itself no longer grades, so `[eval]` is loaded but unused there — leaving it configured is the simplest path. The judge_backend must reference an enabled `[[backends]]` entry. The backend must be OpenAI-compatible — the judge is invoked via the OpenAI Python SDK with `response_format=json_schema` so the verdict shape is enforced at decode time. llama-server, Anthropic's OpenAI-compat endpoint, and OpenAI itself all qualify.

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

Using the same backend/model for both the agent and the judge is allowed but discouraged for pairwise — it's the canonical setup for self-preference bias. For thesis-grade results, the judge should be a different model family from the agent (e.g. agent = local Gemma, judge = remote Claude).

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

By default the runner is single-flight (`--concurrency 1`) — one row at a time. With `--concurrency N` it runs N rows in parallel via a thread pool, which is useful because each row is dominated by network I/O on the agent (local llama-server).

Important caveat: a local llama-server only delivers parallelism when started with `--parallel N` (server slots) — otherwise extra concurrent requests just queue inside the server and you get no speedup. Continuous batching is enabled by default in llama-server, so once the slots are there the GPU schedules requests automatically. Set `N` to match (or exceed) the eval `--concurrency`.

Row order in the output JSON is always sorted by `id`, regardless of completion order.

## Output

Console: progress line per question with a `tools=[name,name,...]` summary, then retrieval / tool-call totals.

JSON file (default `results/YYYYMMDD_HHMMSS.json`):

```json
{
  "schema_version": 3,
  "timestamp": "...",
  "run_id": "a1b2c3d4",
  "server_base_url": "http://127.0.0.1:3000",
  "agent_backend": "local-llama",
  "agent_model": "gemma4-e4b",
  "aggregate": {
    "total": 85,
    "answered": 83,
    "errored": 2,
    "retrieval": {
      "recall_at_5":  {"count": 75, "mean": 0.777, "min": 0.0, "max": 1.0},
      "recall_at_10": {"count": 75, "mean": 0.838, "min": 0.0, "max": 1.0},
      "mrr":          {"count": 75, "mean": 0.609, "min": 0.0, "max": 1.0}
    },
    "tool_calls": {
      "total_invocations": 142,
      "rows_using_tool": 71,
      "by_tool": {
        "semantic_search": {"invocations": 85, "rows": 84},
        "read_document":   {"invocations": 57, "rows": 31}
      }
    }
  },
  "rows": [
    {
      "id": 1,
      "category": "course",
      "difficulty": "easy",
      "question": "...",
      "agent_answer": "...",
      "tool_calls": [
        {"name": "semantic_search", "args": "{\"query\":\"...\"}", "duration_ms": 412},
        {"name": "read_document",   "args": "{\"url\":\"...\"}",   "duration_ms": 89}
      ],
      "retrieved_urls": ["https://...", "..."],
      "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
      "errors": []
    }
  ]
}
```

The `errors` field on each row is a list (not a single string) so a row that hits both a transient retrieval failure and an agent failure retains both messages, each prefixed with its stage (`"retrieval: ..."`, `"agent: ..."`).

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
- `golden_urls` — list of canonical URLs that should be used as ground truth. For cross-doc questions, list all URLs whose chunks should appear in retrieval (eval Step 2 will compute Recall@k against this list). Must be empty when `expected_behavior == "refuse"`.
- `expected_behavior` — `"answer"` (default) when the corpus contains the answer; `"refuse"` when the question is deliberately out-of-corpus and the system should acknowledge it doesn't know. The pairwise judge applies a different rubric for `"refuse"` rows.
- `key_facts` — retained for human review and curation; **not** passed to any judge.
- `notes` — curator-only; ignored by the runner and pairwise judge.

Legacy rows that use `source_url` (single string) are auto-promoted to `golden_urls = [that_url]` at load time, but new rows should use the canonical shape above.

## Extending

- **More questions**: append to `dataset.jsonl`. IDs must be unique.
- **Different pairwise-judge model**: edit `[eval].judge_backend` / `[eval].judge_model` in `config/config.toml`.
- **Different pairwise rubric**: edit `JUDGE_SYSTEM_*` in `etsi_eval/pairwise.py`.

## Pairwise preference

`etsi_eval.pairwise_runner` compares two saved result JSONs head-to-head: same
question, same reference document(s), two candidate answers. The judge picks
which is better (`A` / `B` / `tie`). Each pair is judged TWICE with answer
order swapped — a verdict only counts when both passes agree on the same
underlying answer; positional flips are tagged `unstable` and excluded from
the win-rate. This is the canonical setup for LMSYS-style preference eval
(see Zheng et al., MT-Bench).

Workflow:

```fish
# 1. Run the agent eval before the change
uv run python -m etsi_eval.runner --out results/before.json

# 2. Apply your change (config edit, code change, model swap, ...)

# 3. Run again
uv run python -m etsi_eval.runner --out results/after.json

# 4. Pairwise comparison
uv run python -m etsi_eval.pairwise_runner \
    --a results/before.json --b results/after.json \
    --label-a baseline --label-b after
```

Pairs that can't be fairly judged are skipped and reported (different
`golden_urls`, mismatched `expected_behavior`, empty answer on one side, or
errors recorded on either side).

Output (`pairwise_results/TIMESTAMP.json`) per row includes:

- `winner`: `A` / `B` / `tie` / `unstable` — in the *original* frame
  (i.e. relative to the answer ordering you passed)
- `forward_winner` / `swapped_winner`: raw position-frame verdicts from
  each judge call, for debugging positional bias
- `forward_rationale` / `swapped_rationale`: the judge's reasoning per pass
- `judge_*_tokens`: usage block aggregated across both passes

Aggregate includes `wins_a`, `wins_b`, `ties`, `unstable`, `win_rate_a`
(decisive verdicts only), and a two-sided binomial p-value vs the 50/50
null. Per-category and per-difficulty breakdowns are reported alongside.

The judge backend is the same one configured in `[eval]`. Each pair costs
2× a normal judge call, but the extra cost buys positional-bias control —
see `docs/eval-methodology.md` §2.

## Tests

```fish
uv run pytest
```

Just the parsing/dataset smoke tests — no network.
