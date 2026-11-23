# Retrieval reference

Reference for the retrieval algorithms used by the RAG pipeline, their
parameters, and the academic sources backing them. For *why* hybrid retrieval
was adopted (motivation, eval results, regressions), see
[ADR-008](decisions/008-hybrid-retrieval.md).

## Overview

`Retriever.Retrieve(ctx, query, k)` (`internal/rag/retriever.go`) runs in one of
two modes selected by `[retrieval] retrieval_mode`:

| Mode     | What it does                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------- |
| `dense`  | Cosine similarity between the query embedding and each chunk embedding. Default.                          |
| `hybrid` | Dense (cosine) + sparse (BM25 over an FTS5 inverted index), merged by Reciprocal Rank Fusion. See ADR-008. |

The contract above the retriever is identical in both modes: `Retrieve` returns
the top-K `ScoredChunk`s ordered by score (higher = better).

## Dense retrieval

The query is embedded with the configured `[embedding]` backend, then
`store.SearchSimilar` (`internal/store/chunks.go`) scores every chunk by
cosine similarity:

```
cos(q, c) = (q · c) / (‖q‖ · ‖c‖)
```

Scoring uses a warmed in-memory cache of float32 vectors so each query is a
brute-force O(N · d) dot-product loop in Go (no SQLite scan during the hot
path). At low-thousands chunks this is sub-millisecond.

Cosine is bounded ∈ [-1, 1]; in practice all values are ∈ [0, 1] because
embeddings are non-negative-leaning for natural-language inputs.

## BM25 (sparse leg)

BM25 is a probabilistic ranking function that scores a chunk for a query
based on **term frequency**, **inverse document frequency** (IDF), and
**length normalisation**. For a single-column FTS5 table its score is:

```
BM25(d, q) = Σ_{t ∈ q}  IDF(t) ·  ( f(t,d) · (k1 + 1) ) / ( f(t,d) + k1 · (1 - b + b · |d| / avg|d|) )
```

where:

- `f(t, d)` — number of times term `t` appears in chunk `d`
- `|d|` — length of chunk `d` in tokens
- `avg|d|` — mean chunk length across the corpus
- `IDF(t)` — `log( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )`, with `N` total
  chunks and `n(t)` chunks containing `t`
- `k1` — saturation constant for term frequency (FTS5 default `1.2`)
- `b` — length-normalisation strength, ∈ [0, 1] (FTS5 default `0.75`)

Intuition: `k1` controls how quickly the contribution of repeated occurrences
of the same term saturates; `b` controls whether long chunks get penalised
(`b=0` no penalty; `b=1` full normalisation by length).

### Implementation

- **Engine.** SQLite FTS5 virtual table `chunks_fts` declared with
  `content='chunks', content_rowid='id'` (external content) so the index
  references rows in the `chunks` table without duplicating bytes. AFTER
  INSERT/UPDATE/DELETE triggers on `chunks` keep the index in sync. See
  `internal/store/db.go`. There is no idempotent backfill: chunks that
  predate the FTS5 table are not retroactively indexed — populate an
  existing corpus via the admin **Reprocess** flow (which re-ingests
  every `fetched` `crawl_page` through `ingester.Ingest`, replaying the
  archived bytes from RustFS without re-fetching).
- **Score function.** SQLite's built-in `bm25()` aggregator. It implements
  the BM25F generalisation; for a single column this reduces to standard
  BM25 with `k1 = 1.2, b = 0.75` (the canonical Robertson defaults).
- **Sign convention.** FTS5 returns *lower* values for better matches.
  `SearchKeyword` (`internal/store/chunks.go`) negates the result so the
  "higher score = better" invariant matches `SearchSimilar`.
- **Tokenizer.** `unicode61 remove_diacritics 2` — Unicode-aware, lower-cases,
  splits on non-alphanumeric runes, folds Spanish accents (`María` ↔
  `maria`). No stemmer, no stop-word list (short factoid queries lose recall
  when stop-style tokens are stripped).
- **Query sanitisation.** `sanitizeFTS5Query` replaces every
  non-alphanumeric Unicode rune with a space, wraps each surviving token
  in double quotes and joins them with ` OR `. This (a) splits the query
  the same way the tokenizer indexes, (b) defuses FTS5 operator characters
  (`*`, `"`, `:`, `^`, …) that the user might paste, (c) turns the
  operator words `AND`, `OR`, `NOT` and `NEAR` into plain terms (an agent
  writing `matrícula OR plazo` otherwise produced an FTS5 syntax error),
  and (d) maximises recall — BM25 handles ranking. Quoted tokens still go
  through the tokenizer, so case and accent folding apply.

### Sources

- Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., &
  Gatford, M. (1995). *Okapi at TREC-3.* In *Overview of the Third Text
  REtrieval Conference (TREC-3)*, pp. 109–126. NIST Special Publication
  500-225. The original Okapi BM25 paper. Online:
  <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>.
- Robertson, S., & Zaragoza, H. (2009). *The probabilistic relevance
  framework: BM25 and beyond.* Foundations and Trends in Information
  Retrieval, 3(4), 333–389.
  doi:[10.1561/1500000019](https://doi.org/10.1561/1500000019). The
  canonical reference for the modern BM25 / BM25F formulation and parameter
  defaults.
- SQLite FTS5 documentation, *The bm25() Function*:
  <https://www.sqlite.org/fts5.html#the_bm25_function>.

## Reciprocal Rank Fusion (merge step)

RRF combines two ranked lists into one by summing reciprocal ranks. For a
chunk `d` appearing at rank `r_i` (1-indexed) in retriever `i`:

```
RRF(d) = Σ_i  1 / (k + r_i(d))
```

Retrievers in which `d` does not appear contribute zero. Chunks present in
**both** lists score above chunks present in only one — that's the consensus
boost that makes RRF "agreement-aware".

### Why RRF specifically

- **No score normalisation.** Cosine ∈ [0, 1] and (negated) BM25 ∈ ℝ never
  have to be brought onto a common scale. RRF uses positional rank only.
- **One hyperparameter (`k`).** No per-corpus α tuning between dense and
  sparse weights.
- **Empirically strong.** Cormack et al. 2009 showed RRF outperforms
  Condorcet voting and learned rank fusion on TREC datasets.
- **Cheap.** O(N · m) for `N` candidates and `m` retrievers; on `N ≤ 60`
  per query it's effectively free.

### The `k` constant

`k = 60` is the value Cormack et al. used and recommended. Larger `k`
flattens the contribution from top ranks (low ranks become relatively more
influential); smaller `k` makes the rank-1 hit dominate. 60 is the de facto
default across production RAG systems.

### Candidate-pool depth

Each leg runs at `perList = max(top_k * 3, 30)` so RRF has enough candidates
to fuse before the merger trims back to `top_k`. The floor of 30 was chosen
from the eval-D1 finding that 15 of 19 missed gold chunks in dense
retrieval ranked at positions 7–12 in top-30 — a 30-deep pool covers them
comfortably. See `internal/rag/retriever.go` and ADR-008.

### Implementation

`internal/rag/rrf.go::FuseRRF`. The merger walks both ranked inputs once,
accumulating `1/(k + rank)` per chunk into a map, then sorts the union by
descending fused score and returns the top-K.

### Source

- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank
  Fusion outperforms Condorcet and individual Rank Learning Methods.* In
  *Proceedings of the 32nd International ACM SIGIR Conference on Research
  and Development in Information Retrieval (SIGIR '09)*, pp. 758–759.
  doi:[10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).

## Reranking (optional)

When `[reranker] enabled = true`, `Retriever.Retrieve` over-fetches a wider
candidate pool and re-scores it with a cross-encoder before trimming to
`retrieval.top_k`. This is the "ranking, not visibility" lever from ADR-007
applied at the chunk level: more candidates considered, fewer chunks reaching
the LLM, better facts in the prompt.

### Pipeline position

```
query → embed → dense | hybrid → rerank → trim to top_k → BuildContext → agent
                                  ^
                                  candidate pool sized to max(reranker.top_n, dense/hybrid floor)
```

### Backend constraint

llama-server only. The `/rerank` endpoint is a llama.cpp / Jina extension
(see <https://github.com/ggml-org/llama.cpp/pull/9510>); it is not part of
the OpenAI-compatible surface, so non-llama.cpp backends are rejected at
config-load time and at admin save time. Models loaded with
`--embedding --pooling rank` (e.g. `bge-reranker-v2-m3`,
`Qwen3-Reranker-2B`) are supported; the router build at
`localhost:1337` auto-loads the reranker model on the first request.

### Candidate text

With `title_in_text` (default on), each candidate is sent as
`document title > breadcrumb:\nbody`, the text the index embeds when
`chunking.title_in_search_text` is on; otherwise the body alone. A chunk
rarely names its own document, so without the title the cross-encoder
cannot tell a degree's "150 plazas" chunk from another degree's.

### Score convention

`SearchResult.Score` after rerank carries the cross-encoder's raw relevance
logit. For Qwen3-style rerankers this is a small magnitude (e.g. `8.6e-22`)
because the score is pre-softmax. **Ordering is the load-bearing
property**; the absolute value is preserved as-is for
debugging and observability — no normalisation, no per-batch min-max
scaling. Downstream code that expects "0..1" cosine-style scores has been
audited; only `SourceReference.Score` surfaces this number, and it is
displayed verbatim.

### Token-budget guard

Cross-encoders concatenate `(query, doc)` into a single sequence — both
must fit the model's context window. Before sending each batch,
`RerankClient.fitToContext`:

1. Tokenises the query via the backend's `/tokenize` endpoint
   (`tokencount.LlamaCppCounter`).
2. Computes the per-doc budget as `max_tokens - len(query_tokens) -
   reserved_tokens`. `reserved_tokens` (default 16) covers BOS / EOS /
   separator overhead.
3. For any document whose token count exceeds the budget, tokenises →
   slices to the budget → detokenises. Truncation is preferred over
   dropping because losing a doc means losing its rank slot.
4. If the query alone overflows the budget, the rerank step is skipped
   and the pre-rerank ordering passes through unchanged.

llama-server evaluates each `(query, doc)` pair in one physical batch, so
start the reranker with `--batch-size` and `--ubatch-size` at least as large
as its context size (e.g. `--ctx-size 8192 -b 8192 -ub 8192`). At the
default 512-token `--ubatch-size`, any pair over 512 tokens makes the whole
`/rerank` call fail with HTTP 500.

### Failure modes (all soft fall-through)

The reranker is opt-in optimisation, never a critical path. On any of:

- network error / timeout
- HTTP non-2xx
- malformed response (missing `results`, out-of-range `index`)
- token-counter unavailable
- query overflows the context budget
- per-request deadline exceeded

…`Retrieve` logs a warn line and returns the un-reranked candidates
trimmed to `top_k`. Chats never fail because the reranker is offline.
The same logic applies when the live `Reranker` is `nil` (e.g. config
re-resolution after an admin save failed).

### Candidate-pool sizing

The reranker only helps when it sees more candidates than the agent
will. `Retrieve` raises the candidate-pool depth to
`reranker.top_n` (configurable, default 30):

- **dense** mode: `pool = max(reranker.top_n, retrieval.top_k)`.
- **hybrid** mode: each leg fetches `perList = max(reranker.top_n, top_k * 3, 30)` and RRF fuses down to `pool` —
  preserves ADR-008's existing 30-candidate floor.

After rerank, the result is trimmed to `retrieval.top_k`.

### Layered safeguards

| Boundary | Check | Reaction |
|---|---|---|
| `config.Validate` (load + admin save) | `enabled` requires non-empty backend, non-empty model, backend exists & enabled, `IsLlamaCpp`, `top_n >= top_k` | Reject; admin UI reverts in-memory config and toasts |
| `Server.resolveRerankClient` | Re-checks the same invariants at runtime | Returns `(nil, error)`; `Retriever.SetReranker(nil, 0)` |
| `Server.attachReranker` | Token-counter availability | Disables reranker, logs warn |
| `Retriever.Retrieve` | `len(candidates) <= 1` | Skips reranker (nothing to reorder) |
| `RerankClient.fitToContext` | Query overflows budget | Returns input ordering, no error |
| `applyRerank` (`internal/rag/retriever.go`) | Network / HTTP / decode error | Returns un-reranked candidates trimmed to `top_k` |

### Sources

- llama.cpp PR #9510, "Reranking endpoint":
  <https://github.com/ggml-org/llama.cpp/pull/9510>.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.*
  arXiv:1901.04085. The canonical reference for cross-encoder reranking
  over candidate sets generated by a first-stage retriever.
  <https://arxiv.org/abs/1901.04085>.
- BAAI bge-reranker family — bilingual / multilingual cross-encoders
  trained for reranking:
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>.

## Code map

| Component                                  | File                                  |
| ------------------------------------------ | ------------------------------------- |
| Cosine search; in-memory vector cache        | `internal/store/chunks.go`, `cache.go` |
| FTS5 schema, triggers, tokenizer            | `internal/store/db.go`                |
| `SearchKeyword`, `sanitizeFTS5Query`        | `internal/store/chunks.go`            |
| `FuseRRF`                                   | `internal/rag/rrf.go`                 |
| `Retrieve` mode switch, candidate sizing    | `internal/rag/retriever.go`            |
| `RetrievalConfig.RetrievalMode` validation  | `internal/config/config.go`           |
| `applyRerank` (rerank step + soft fall-through) | `internal/rag/retriever.go`        |
| `llm.Reranker` interface, `RerankClient`        | `internal/llm/rerank.go`           |
| `RerankerConfig` validation                     | `internal/config/config.go`        |
| `Server.attachReranker`, `resolveRerankClient`  | `internal/server/wiring.go`        |
| Hybrid eval evidence and decision rationale     | `docs/decisions/008-hybrid-retrieval.md` |
| Reranker decision rationale                     | `docs/decisions/009-reranker.md`     |

## See also

- [ADR-007: RAG context budget](decisions/007-rag-context-budget.md) — why
  the right lever is **ranking**, not visibility (wider context regressed
  the small-model agent by 31 points).
- [ADR-008: BM25 + Dense Hybrid via RRF](decisions/008-hybrid-retrieval.md)
  — eval results, regressions analysis, and the case for a cross-encoder
  reranker as the next intervention.
- [ADR-009: Cross-encoder reranker](decisions/009-reranker.md) — the
  reranker stage itself: design, llama.cpp-only constraint, soft
  fall-through behaviour, and (after eval rerun) results.
