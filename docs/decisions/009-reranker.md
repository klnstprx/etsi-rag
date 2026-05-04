# ADR-009: Cross-encoder Reranker over Hybrid Candidates

## Status
Accepted (2026-05-04 — bge-reranker-v2-m3 chosen after eval sweep across
three reranker models and two agent sizes; retrieval metrics improve
unambiguously, judge-graded Overall is flat-to-positive)

## Context

ADR-008 ("BM25 + Dense Hybrid via RRF", 2026-05-04) shipped hybrid retrieval
with a +4.1 pts Overall mean on the 85-question eval, but its follow-up
section documented a non-uniform result: 29 rows improved, **25 rows
regressed**, 31 unchanged. Post-hoc analysis attributed the regressions to
five patterns — wrong-document substitution, refusal weakening, mixed-doc
hallucination, detail loss, and structural cross-doc misses — all driven by
the same root cause:

> "gemma4-e4b cannot reliably disambiguate lexically-adjacent chunks once
> they reach the prompt."

The same diagnosis appeared in ADR-007's wider-context A/B (-31 pts when
top_k went 10 → 20). The lever is consistent across both ADRs:
**small-model RAG is retrieval-precision-bound, not retrieval-coverage-bound.**
The next intervention must suppress lexical near-misses before they reach
the LLM, not surface more candidates.

ADR-008 named the canonical fix:

> "A cross-encoder reranker over the top-30 hybrid candidates… It re-scores
> each (query, chunk) pair jointly rather than via decomposable token
> overlap, so the MUTDE chunk that lexically matches an MUII question gets
> demoted because the cross-encoder sees the *whole* chunk doesn't actually
> answer the question."

## Decision

Add a **cross-encoder reranker** as a pipeline stage between candidate
generation and `BuildContext`.

### Mechanics

- **Backend.** llama-server's `/rerank` endpoint
  ([llama.cpp PR #9510](https://github.com/ggml-org/llama.cpp/pull/9510)),
  Cohere/Jina-style request shape. Verified live against the router build at
  `localhost:1337` with `qwen-reranker` (Qwen3-VL-Reranker-2B.Q8_0,
  n_ctx=4096). Both `/rerank` and `/v1/rerank` paths return 200; the SDK is
  configured to land on `/v1/rerank` via `option.WithBaseURL`.
- **Client.** `internal/llm/rerank.go::RerankClient`, built on
  `openai.Client.Post(ctx, "rerank", ...)` — the SDK's documented escape
  hatch — so the auth / retry / base-URL machinery is shared with
  `EmbedClient` / `ChatClient`. No raw `net/http`. The non-OpenAI endpoint
  is the only divergence from the existing pattern.
- **Pipeline integration.** `Pipeline.Retrieve` over-fetches a wider
  candidate pool (`max(reranker.top_n, top_k * 3, 30)` for hybrid;
  `max(reranker.top_n, top_k)` for dense), reranks, and trims to `top_k`.
  Image markers (`[[image:ref]]`) are stripped before sending — same
  hygiene the embed path already uses (`Pipeline.Ingest`).
- **Token-budget guard.** Cross-encoders concatenate (query, doc) into a
  single sequence; both must fit `n_ctx`. `RerankClient.fitToContext`
  tokenises the query, computes a per-doc budget of
  `max_tokens - len(query) - reserved`, and truncates oversized docs via
  `Tokenize → slice → Detokenize` against the existing
  `tokencount.LlamaCppCounter`. Truncation is preferred over dropping
  because losing a doc means losing its rank slot.
- **Score convention.** `SearchResult.Score` after rerank carries the
  cross-encoder's raw relevance logit. For Qwen3 rerankers this is a small
  magnitude (e.g. `8.6e-22`) because the score is pre-softmax. **Ordering
  is the load-bearing property**; the absolute value passes through as-is
  for debugging and observability — no normalisation, no min-max scaling.
- **Llama-cpp only.** `/rerank` is a llama.cpp / Jina extension, not an
  OpenAI endpoint. `config.Validate` rejects `[reranker] enabled = true`
  against any backend with `is_llama_cpp = false`, with an error message
  naming the offending backend.
- **Opt-in.** `[reranker] enabled = false` by default. The same hot-rebuild
  pattern as `retrieval_mode` (ADR-008): toggling `enabled` in the admin UI
  rebuilds the client without a server restart.

### Soft fall-through (load-bearing)

The reranker is opt-in optimisation, never a critical path. Any of the
following degrade silently to the un-reranked candidates:

- network error / timeout
- HTTP non-2xx (model unloaded, model unavailable, OOM)
- malformed response (missing `results`, out-of-range `index`)
- query overflows the cross-encoder's context window
- token-counter unavailable
- per-request deadline exceeded

`Pipeline.applyRerank` logs a warn line and returns the pre-rerank
ordering trimmed to `top_k`. **Chats never fail because the reranker is
offline.** This is the same design choice the embedding stage made:
`Pipeline.Retrieve` returns errors only for hard failures (no embedder,
SQL error); transient backend issues degrade the result rather than
the request.

### Layered safeguards

`Validate` and `resolveRerankClient` both enforce the llama-cpp
constraint — duplicated intentionally because `Validate` can be bypassed
by editing TOML on disk. The runtime resolver catches that without
crashing the server.

| Boundary | Check | Reaction |
|---|---|---|
| `config.Validate` | enabled requires backend / model / `IsLlamaCpp` / `top_n >= top_k` | Reject; admin UI reverts in-memory config |
| `Server.resolveRerankClient` | re-checks the same invariants | `(nil, error)` → `Pipeline.SetReranker(nil, 0)` |
| `Server.attachReranker` | counter availability | Disables reranker, logs warn |
| `Pipeline.Retrieve` | nil reranker, single candidate | Skips rerank silently |
| `RerankClient.fitToContext` | query overflows budget | Returns input ordering, no error |
| `Pipeline.applyRerank` | request error | Returns un-reranked candidates |

### Auto-detection

`probeLlamaCppBackends` is extended with the reranker stage. The probe
calls `/props?model=<reranker-model>`, which the router uses both to
report `n_ctx` and as a side-effect to load the reranker model — so the
first user `/rerank` call avoids cold-load latency. The probed `n_ctx`
becomes `cfg.Reranker.MaxTokens` if not explicitly configured.

## Evidence

Four eval runs were executed across the design space. Configuration held
constant: `retrieval_mode = "hybrid"`, judge = remote-openai / gpt-5.4-mini,
85-question dataset. What varied: reranker model, agent model, corpus
state.

### Eval-methodology caveat

The judge-graded Overall score is an LLM-as-judge metric, dependent on
judge choice and per-call non-determinism (typical per-row variance ±5–15
points). The retrieval-tier metrics (Recall@k, MRR) are objective
set-membership computations against ground-truth gold URLs and depend on
no LLM. **The retrieval metrics are the load-bearing comparison;
LLM-judged Overall is reported as a secondary trend signal.** Limitations
section at end.

### Sweep summary (apples-to-apples, n=83 LLM-scored / n=73 retrieval after corpus rebuild)

| Run                                        | Overall | Recall@5 | Recall@10 | MRR | Δ Overall vs hybrid |
|--------------------------------------------|---------|----------|-----------|-----|---------------------|
| Baseline: dense + e4b                      | 76.88   | 0.725    | 0.808     | 0.560 | −3.94 |
| Hybrid + e4b (ADR-008)                     | 80.82   | 0.743    | 0.811     | 0.591 | — |
| Hybrid + qwen-reranker (0.6B) + e4b        | (n=59 only — see below) | — | — | — | — |
| Hybrid + Qwen3-VL-Reranker-2B + e4b        | (n=59 only — see below) | — | — | — | — |
| **Hybrid + bge-reranker-v2-m3 + e4b**      | 80.52   | **0.763** | **0.838** | **0.607** | −0.30 |
| **Hybrid + bge-reranker-v2-m3 + 26b-a4b**  | **81.04** | **0.777** | 0.838    | 0.609 | **+0.22** |

The two qwen reranker runs predate the corpus rebuild and could only be
evaluated on n=59 (26 rows errored due to "references not found in DB",
diagnosed as a separate corpus issue, not a reranker failure — see
"corpus drift" below). On their respective subsets they regressed Overall
by −1.5 (0.6B) and −1.8 (2B) versus hybrid alone, both with high per-row
variance. Detailed per-row analysis attributed the regressions to two
patterns: (a) confident displacement of gold chunks out of the top-5 by
the smaller reranker, and (b) refusal-weakening on out-of-corpus
questions when the small agent received tighter chunks.

bge-reranker-v2-m3 (BAAI, 568M params, multilingual cross-encoder) is the
chosen model. Compared to both qwen variants it has lower variance,
matches the course-category gain (+10.5 vs hybrid on n=12), and is the
only reranker run where Recall@5, Recall@10 and MRR all improve cleanly
versus hybrid alone.

### bge-m3 + e4b vs hybrid + e4b — per-category (same retrieval pipeline, agent held constant)

| Category   | n  | Hybrid  | bge-m3  | Δ |
|------------|----|---------|---------|----|
| course     | 12 | 76.9    | **87.4** | **+10.5** ✓✓ |
| cross-doc  | 10 | 55.5    | 58.5    | +3.0 ✓ |
| degree     | 24 | 85.5    | 84.5    | −1.1 |
| general    | 12 | 90.7    | 86.1    | −4.6 |
| procedure  | 25 | 83.6    | 79.6    | −4.0 |

Win/loss vs hybrid: 17 / 19 / 47 same (Δ > ±5 threshold), mean Δ = −0.30.

**Course category fix is robust.** The canonical "lexically-similar wrong
PDF" failure pattern (e.g. row 5: hybrid retrieves course 51181's syllabus
while the question asks about course 51175 — both share the prefix
`5103_AsigUMA_`) is consistently corrected. Row 5 went 10 → 85 across all
reranker variants.

**Refusal-weakening is the dominant new failure pattern.** Row 84 ("¿Cuándo
fue fundada la UMA?", a refuse-mode question with no gold in the corpus)
collapsed 100 → 15 with bge-m3 + e4b. The reranked candidate set looked
authoritative-enough to gemma4-e4b that it synthesised related-but-wrong
historical dates instead of refusing. Hybrid's noisier candidates had let
the same agent recognise the topical mismatch and refuse.

### Effect of agent capacity (bge-m3 reranker held constant, e4b → gemma4-26b-a4b)

| Metric   | bge-m3 + e4b | bge-m3 + 26b-a4b | Δ (agent effect) |
|----------|--------------|------------------|------------------|
| Overall  | 80.52        | **81.04**        | +0.52 |
| Recall@5 | 0.763        | 0.777            | +0.014 |
| MRR      | 0.607        | 0.609            | +0.002 |

Per-row: 21 wins, 19 losses, 43 same, mean Δ = +0.52. Retrieval is nearly
identical between agent sizes (same retrieval pipeline; tiny differences
come from slightly different agent-issued search queries). The aggregate
lift is small but lands on specific failure patterns:

- **Refusal weakening reverses on the larger agent.** Row 84 went
  15 → 95 with the same retrieved chunks. The 26b-a4b agent recognised
  the chunks didn't actually answer the founding-date question and
  refused appropriately.
- **Procedure category gains +2.0** under the larger agent — the
  category where the small agent had been losing −4.0 from rerank.
- **Variance remains high.** Largest 26b-a4b regression is row 34
  (general, e4b 72 → 26b 5), inspected manually: same retrieved URLs
  for both agents, neither retrieved the gold direction page, both
  agents fabricated different director names from the same imperfect
  candidate set. The 67-point gap is judge artefact, not a real
  quality difference; both answers are equally wrong against the gold
  reference.

### Selected per-row evidence

Reranker wins (bge-m3 + 26b vs hybrid + e4b):
- row 5 (course): 10 → 72 (+62) — wrong-PDF demoted
- row 61 (procedure): 10 → 90 (+80)
- row 19 (cross-doc): 55 → 92 (+37)
- row 71 (cross-doc): 55 → 78 (+23)

Cross-doc gains (rows 19, 71) are notable: ADR-008 had marked structural
cross-doc questions as "won't help" and capped at 55 mean. With reranking
plus the larger agent, two of the ten cross-doc rows clearly improved.

### Aggregate decision

The Overall judge-graded score being flat (−0.30 with e4b agent, +0.22
with 26b-a4b) is at the noise floor of LLM-as-judge eval. The
retrieval-tier signal is unambiguous:

- Recall@5: **+0.034** vs hybrid (0.743 → 0.777 with 26b-a4b agent)
- Recall@10: **+0.027** vs hybrid (0.811 → 0.838)
- MRR: **+0.018** vs hybrid (0.591 → 0.609)

Set-membership against gold URLs is independent of judge model and
reproducible across reruns. **The reranker improves retrieval; the
absorbed judge-graded answer-quality is consistent with documented
small-model limitations (refusal stability, hallucination from confident
retrievals) rather than with the reranker degrading the pipeline.**

### Corpus drift (auxiliary fix)

Three pre-bge-m3 reruns errored 26/85 rows with "references not found in
DB". Diagnosed as two distinct bugs:

- **Cache dim mismatch.** `embeddingCache.appendChunks` returned an error
  when the active embedding dim differed from the warmed dim (e.g. after
  swapping embedding models without `make clean`). Fixed in
  `internal/store/cache.go`: dim mismatch now invalidates the cache and
  lets `SearchSimilar` fall back to SQL until the next Warm.
- **Content-hash dedup race.** UMA's CMS serves identical extracted
  content at the bare program URL (`/grado-en-X/`) and at one of its
  `/cms/menu/...` sub-pages. The crawler's content-addressed dedup
  stamps `documents.url` with whichever URL won the timing race —
  non-deterministic across crawls. Fixed in
  `internal/crawler/crawler.go::preferCanonical` and the embed-stage
  dedup branch: when a content-hash collision surfaces a URL with fewer
  path segments than the existing one, swap `documents.url` to the
  more canonical form.

After both fixes and a `make clean` + recrawl, only 2/85 rows error (a
different URL pattern not addressed by the canonical-swap heuristic).
n=83 is the apples-to-apples row count used throughout this section.

## Limitations of the eval methodology

- **LLM-as-judge dependency.** The Overall, Faithfulness, Correctness
  and Completeness scores are produced by an LLM (gpt-5.4-mini) reading
  each (question, answer, reference) tuple and emitting 0–100 scalars.
  Per-call non-determinism, judge bias toward fluent-sounding answers,
  and prompt-phrasing sensitivity all introduce variance that is not
  captured by the reported means.
- **Reference-document mismatch.** Some eval rows have a single labelled
  golden URL that may not be the only correct answer in the corpus
  (e.g. row 34's director question — the eval expects one specific
  page, but multiple corpus pages contain plausibly-correct director
  information). Strict reference-matching by the judge produces score
  variance unrelated to true answer quality.
- **No human calibration.** Per-row judge scores are not validated
  against human annotation. A future improvement is to hand-annotate
  a 20–30 row sub-sample and report inter-rater agreement (Cohen's κ)
  between author and judge.
- **No pairwise preference.** Absolute 0–100 scoring is more brittle
  than pairwise A/B preference judgements. A future improvement is to
  rerun the comparison with the same judge prompted to choose between
  two answers per row and report a win-rate statistic.

These limitations apply equally to ADR-007 and ADR-008's evidence
sections; this ADR is the first to call them out explicitly. The
retrieval-tier metrics are not affected.

## Consequences

- **Storage:** none. The reranker writes nothing.
- **Latency:** added per-query cost is bounded — top-30 cross-encoder
  forward passes on a Qwen3-Reranker-2B run in a few hundred ms on the
  router GPU, well below the agent's per-turn budget. The first call
  after server start carries cold-load latency (model unloaded by
  default in the router); the 30s default `timeout` absorbs this.
- **No new dependencies.** llama-server is already deployed and the
  reranker model is already registered in the router preset.
- **Observability.** `SourceReference.Score` displays raw cross-encoder
  logits when the reranker has run. UI normalisation could be added as
  a follow-up but is intentionally not part of this ADR — the small
  numbers are useful for debugging precisely *because* they're
  unmistakably cross-encoder output, not cosine.
- **Score interpretation.** Downstream code that compared scores across
  queries (none currently) would see incomparable values when reranking
  is on. `Pipeline.applyRerank` documents this in a comment block.

## Rejected alternatives

- **Re-rank with the agent LLM.** Asking gemma4-e4b to score chunks
  against the query is what the agent already does implicitly with the
  in-context chunks — the very behaviour ADR-008's regression analysis
  showed it does badly.
- **Score normalisation (min-max / softmax).** Loses information for no
  practical gain. The UI surfaces a single score per chunk; the user
  doesn't compare across queries. Raw logits are diagnostic.
- **Drop reranker scores, keep cosine/RRF as `SearchResult.Score`.**
  Considered briefly. Rejected because the reranker's score is the
  primary signal that it ran — useful for debugging "did the reranker
  do anything?" without inspecting logs.
- **Multi-stage cascade (lexical → dense → cross-encoder).** Three-tier
  retrieval is overkill at this corpus size. Hybrid (RRF) covers the
  candidate-generation tier; rerank covers the precision tier. One
  more stage adds tuning surface without obvious gain.
- **Reranker as a tool.** The agent doesn't decide *whether* to rerank;
  the pipeline does. Tooling rerank exposes a knob the LLM shouldn't
  have.

## Critical files

- `internal/llm/rerank.go` — `RerankClient`, `Reranker` interface,
  `fitToContext` truncation.
- `internal/llm/llm.go` — `Reranker` and `RerankResult` shapes.
- `internal/rag/pipeline.go` — `applyRerank`, candidate-pool sizing,
  soft fall-through.
- `internal/config/config.go` — `RerankerConfig`, `Validate` rules.
- `internal/server/wiring.go` — `resolveRerankClient`,
  `attachReranker`, `probeLlamaCppBackends` reranker stage.
- `internal/admin/config_handler.go` — admin section + save handler.
- `config/config.default.toml` — default `[reranker]` block.
- `docs/retrieval.md` — Reranking section, code map, score convention.
- `docs/configuration.md` — `[reranker]` reference.
- `docs/decisions/008-hybrid-retrieval.md` — context for *why* a
  reranker is the chosen next intervention.
