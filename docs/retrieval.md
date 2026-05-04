# Retrieval reference

Reference for the retrieval algorithms used by the RAG pipeline, their
parameters, and the academic sources backing them. For *why* hybrid retrieval
was adopted (motivation, eval results, regressions), see
[ADR-008](decisions/008-hybrid-retrieval.md).

## Overview

`Pipeline.Retrieve(ctx, query, k)` (`internal/rag/pipeline.go`) runs in one of
two modes selected by `[retrieval] retrieval_mode`:

| Mode     | What it does                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------- |
| `dense`  | Cosine similarity between the query embedding and each chunk embedding. Default.                          |
| `hybrid` | Dense (cosine) + sparse (BM25 over an FTS5 inverted index), merged by Reciprocal Rank Fusion. See ADR-008. |

The contract above the pipeline is identical in both modes: `Retrieve` returns
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
  `internal/store/db.go`.
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
  non-alphanumeric Unicode rune with a space, then joins the surviving
  tokens with ` OR `. This (a) splits the query the same way the tokenizer
  indexes, (b) defuses FTS5 operator characters (`*`, `"`, `:`, `^`, …)
  that the user might paste, and (c) maximises recall — BM25 handles
  ranking.

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

Each leg runs at `kEach = max(top_k * 3, 30)` so RRF has enough candidates
to fuse before the merger trims back to `top_k`. The floor of 30 was chosen
from the eval-D1 finding that 15 of 19 missed gold chunks in dense
retrieval ranked at positions 7–12 in top-30 — a 30-deep pool covers them
comfortably. See `internal/rag/pipeline.go` and ADR-008.

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

## Code map

| Component                                  | File                                  |
| ------------------------------------------ | ------------------------------------- |
| Cosine search, in-memory vector cache       | `internal/store/chunks.go`            |
| FTS5 schema, triggers, tokenizer            | `internal/store/db.go`                |
| `SearchKeyword`, `sanitizeFTS5Query`        | `internal/store/chunks.go`            |
| `FuseRRF`                                   | `internal/rag/rrf.go`                 |
| `Retrieve` mode switch, candidate sizing    | `internal/rag/pipeline.go`            |
| `RetrievalConfig.RetrievalMode` validation  | `internal/config/config.go`           |
| Eval evidence and decision rationale        | `docs/decisions/008-hybrid-retrieval.md` |

## See also

- [ADR-007: RAG context budget](decisions/007-rag-context-budget.md) — why
  the right lever is **ranking**, not visibility (wider context regressed
  the small-model agent by 31 points).
- [ADR-008: BM25 + Dense Hybrid via RRF](decisions/008-hybrid-retrieval.md)
  — eval results, regressions analysis, and the case for a cross-encoder
  reranker as the next intervention.
