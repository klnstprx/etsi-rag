# ADR-007: Keep RAG Context Narrow on Small Models

## Status
Accepted

## Context

The eval baseline (2026-05-04, gemma4-e4b agent / gpt-5.4-mini judge,
85 questions) reported Overall mean 76.1, with Recall@5 = 70.6%, Recall@10 =
78.6%, MRR = 0.545. Error analysis (D1 diagnostic, see
`eval/scripts/d1_topk_probe.py`) showed:

- 19/75 answer rows missed at top_k=5.
- 14/19 (74%) of those misses surface in top_k=20, 15/19 (79%) in top_k=30.
- Most failures are gold chunks ranked 7-12 by the dense retriever — they
  *exist*, they just aren't promoted high enough.

Hypothesis: expose the agent to more retrieved chunks (raise top_k) and
let more of them reach the LLM (raise `max_context_tokens`). With chunk
budget at 600 tokens and `max_context_tokens=4092`, only ~6-7 chunks
reached the LLM even though `top_k=10` retrieved 10. The rank 8-12 chunks
where the gold often lived were silently dropped at `BuildContext`.

We ran a controlled A/B:

| Setting               | Baseline | Bumped |
|-----------------------|----------|--------|
| `[rag].top_k`         | 10       | **20** |
| `[rag].max_context_tokens` | 4092 | **8192** |
| Other config          | unchanged | unchanged |

## Decision

**Revert. Keep `top_k=10` and `max_context_tokens=4092`.** Address
retrieval quality at the **ranking** level (next: BM25 + dense hybrid via
RRF) rather than by widening the context window.

## Evidence

Both runs are committed under `eval/results/`:

- `baseline_topk10_ctx4092.json` — Overall **76.1**, F=85.2, C=85.0, Cm=81.6
- `regression_topk20_ctx8192.json` — Overall **45.5**, F=68.6, C=62.3, Cm=51.7

A 31-point Overall regression. Recall@5/@10/MRR were unchanged across runs
(as expected — the eval probe queries `/v1/rag/query` directly with its own
top_k=10 and is independent of `[rag].top_k`). The metrics that moved are
the judge dimensions, which reflect what the agent could actually *use*.

The regression hit rows that previously scored 90-100. Concretely, **#17**
("¿Quién preside la Comisión Académica y de Calidad de la ETSI
Informática?"):

- Baseline (6-7 chunks visible): correctly named *Manuel Nicolás Enciso
  García-Oliveros*, the ETSI's chair → Overall 100.
- Bumped (13+ chunks visible): the chunks now also included the Comisión
  Académica del **Doctorado en Tecnologías Informáticas** (a different,
  semantically-adjacent body chaired by Llanos Mora López). The agent
  picked the wrong one and confidently cited the doctorate's resolution
  → Overall 0.

The pattern repeats across ~15 rows: gemma-4-e4b loses focus when the
context contains multiple semantically-similar but topically-distinct
chunks. This is the textbook "lost-in-the-middle" / distractor failure
of small models.

## Consequences

- **Constraint documented.** Wider context is not free. Any future
  proposal to increase `max_context_tokens` or `top_k` for this agent must
  be accompanied by either (a) a stronger model, or (b) a reranker that
  filters distractors before BuildContext, or (c) evidence from a fresh
  controlled A/B.
- **Default tool description is neutral.** The `semantic_search` tool's
  `top_k` parameter description does not advertise the configured default
  to the agent, since prompting the agent to request a higher top_k would
  reproduce this failure mode.
- **Path forward.** The right lever is precision over recall: get the
  correct chunk to rank #1-#5 instead of #7-#12. Hybrid retrieval (BM25 +
  dense, combined via reciprocal-rank fusion) is the next D-step.
  Especially well-suited to this corpus because the failing queries hinge
  on exact tokens (`Plazas: 140`, `secretariaetsii@uma.es`,
  `952 13 14 95`) where dense embeddings systematically under-rank.
- **Memoria angle.** This negative result is publishable in §7.5 / §7.6
  as a controlled demonstration that naïvely expanding RAG context
  degrades small-model performance — pre-justifies the investment in
  hybrid retrieval.

## Rejected alternatives

- **Sweep for a sweet spot** (top_k 12–15, ctx 5000–6000). The shape of
  the regression (already-correct rows breaking) suggests there is no
  flat plateau where the model tolerates extra distractors gracefully;
  any additional irrelevant chunk introduces a coin-flip risk. Worth ~$0.40
  of judge calls if a future curator wants the data point, but the
  qualitative finding is already decisive.
- **Bump only `max_context_tokens` (not `top_k`).** Same root cause —
  rank 7-10 chunks would now reach the LLM, polluting the prompt with
  the same kind of distractor. Less severe than the full bump but same
  failure mode.
- **Increase `[agent].context_window`.** Doesn't help: the bottleneck is
  not the model's max input length (gemma4-e4b is loaded with 32k); it's
  the model's ability to *use* a long input correctly.
