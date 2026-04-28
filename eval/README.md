# etsi-eval

LLM-as-judge evaluation harness for the etsi-rag agent.

## How it works

For each question in `dataset.jsonl`:

1. Look up the reference document(s) in `data/etsi-rag.db` by each entry in `golden_urls` (read-only). Skipped for `expected_behavior == "refuse"` rows.
2. Probe the retrieval layer via `POST /v1/rag/query` to compute Recall@k / MRR against `golden_urls` (URL-level, k = 5 and 10). Skipped for refuse rows where the metric is undefined.
3. Ask the agent via `POST /v1/chat/completions` — single user turn.
4. Feed `(question, agent answer, reference docs, expected_behavior)` to the **judge LLM** (configured separately in `[eval]` — see "Configuring the judge" below; called directly at its `/chat/completions` endpoint so the judge never goes through the agent's tool loop).
5. The judge returns three independent sub-scores (see "Rubric" below). For refuse rows, the rubric scores how well the model declined to answer.
6. Results are aggregated per dimension, retrieval metric, category, and difficulty and written to `results/TIMESTAMP.json`.

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

## Configuring the judge

A `[eval]` section is **required** in `config/config.toml`. The judge_backend must reference an enabled `[[backends]]` entry. The eval tool will refuse to start if the section is missing or malformed.

```toml
[[backends]]
name = "remote-claude"
base_url = "https://api.anthropic.com/v1"
api_key = "sk-..."
enabled = true

[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"
```

Using the same backend/model for both the agent and the judge is allowed but discouraged — it's the canonical setup for self-preference bias and the runner will print a warning. For thesis-grade results, the judge should be a different model family from the agent (e.g. agent = local Gemma, judge = remote Claude).

## Prerequisites

- Python 3.11+ (uses `tomllib` from stdlib).
- `uv` on PATH.
- A running etsi-rag server (default `http://127.0.0.1:3000`).
- A populated `data/etsi-rag.db` (crawl has been run at least once).
- The backend referenced by `[agent].backend` in `config/config.toml` must be up.

## Run

```fish
cd etsi-rag/eval
uv sync

# full 20-question run
uv run python -m etsi_eval.runner

# filters
uv run python -m etsi_eval.runner --difficulty easy
uv run python -m etsi_eval.runner --category course
uv run python -m etsi_eval.runner --ids 1,7,12

# retrieval probe top_k (default 10)
uv run python -m etsi_eval.runner --retrieval-top-k 20

# explicit output path
uv run python -m etsi_eval.runner --out /tmp/run1.json
```

## Output

Console: progress line per question, then a summary table with per-category / per-difficulty means.

JSON file (default `results/YYYYMMDD_HHMMSS.json`):

```json
{
  "timestamp": "...",
  "server_base_url": "http://127.0.0.1:3000",
  "agent_backend": "local-llama",
  "agent_model": "agent-model",
  "judge_backend": "remote-claude",
  "judge_model": "claude-sonnet-4-6",
  "aggregate": {
    "total": 20,
    "scored": 20,
    "by_dimension": {
      "faithfulness": {"count": 20, "mean": 82.1, "min": 30, "max": 100},
      "correctness":  {"count": 20, "mean": 78.4, "min": 20, "max": 100},
      "completeness": {"count": 20, "mean": 70.5, "min": 0,  "max": 100},
      "overall":      {"count": 20, "mean": 65.2, "min": 0,  "max": 95}
    },
    "retrieval": {
      "recall_at_5":  {"count": 18, "mean": 0.722, "min": 0.0, "max": 1.0},
      "recall_at_10": {"count": 18, "mean": 0.833, "min": 0.0, "max": 1.0},
      "mrr":          {"count": 18, "mean": 0.611, "min": 0.0, "max": 1.0}
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
      ...
    }
  ]
}
```

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
