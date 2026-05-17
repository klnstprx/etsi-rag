# ADR-008: BM25 + Dense Hybrid Retrieval via RRF

## Status
Accepted (2026-05-04 — evaluated end-to-end; +4.1 pts Overall mean)

## Context

The 2026-05-04 baseline eval reported Overall=76.1, Recall@5=70.6%, MRR=0.545
on the 85-question dataset. Error analysis (`eval/scripts/d1_topk_probe.py`)
showed that **15/19 of the Recall@5 misses surface in top-30** of the dense
retriever — the gold chunks exist in the corpus and score nonzero, but rank
at positions 7–12 instead of 1–5.

ADR-007 ruled out widening the context window: a controlled A/B
(top_k=10/ctx=4092 → top_k=20/ctx=8192) regressed Overall by 31 points.
gemma4-e4b is small and gets confused when the prompt includes multiple
semantically-similar but topically-distinct chunks.

The remaining lever is **ranking**: get the right chunk to position 1–5
without changing the context budget. The failing queries hinge on exact
tokens that dense embeddings systematically under-rank — `Plazas: 140`,
`secretariaetsii@uma.es`, `952 13 14 95`, course/programme codes — exactly
where lexical (BM25) matching dominates.

## Decision

Add **SQLite FTS5 BM25** alongside the existing dense retriever and combine
the two via **Reciprocal Rank Fusion (RRF)** with `k=60`.

Mechanics:

- **Index.** Single FTS5 virtual table, `chunks_fts`, declared with
  `content='chunks', content_rowid='id'` (external content) so it indexes
  `chunks.content` without duplicating bytes. AFTER INSERT/UPDATE/DELETE
  triggers on `chunks` keep it in sync. Chunks that predate the FTS5
  table are not retroactively indexed; populate an existing corpus by
  running the admin **Reprocess** flow, which re-ingests every
  `fetched` page from the archived RustFS bytes (no re-fetch).
- **Tokenizer.** `unicode61 remove_diacritics 2` folds Spanish accents
  (`María` ↔ `maria`). No stemmer, no stop words — short factoid queries
  lose signal when stop-style tokens are stripped.
- **Query sanitisation.** User queries are mapped: every non-alphanumeric
  Unicode rune becomes a space, the result is split on whitespace, and the
  remaining tokens are joined with ` OR `. This guarantees no FTS5 syntax
  characters reach the parser, splits hyphenated terms the same way the
  tokenizer indexes them, and keeps recall high — BM25 handles ranking.
- **Score convention.** FTS5 `bm25()` returns *lower* values for better
  matches. `SearchKeyword` stores `-bm25_value` so the
  "higher Score = better" invariant matches `SearchSimilar`.
- **Fusion.** `FuseRRF(dense, sparse, topK, k)` builds a per-chunk fused
  score `Σ 1/(k + rank_i)` with `k=60`. RRF uses positional rank only, so
  it does not require score normalisation between cosine (∈[0,1]) and
  BM25 (∈ℝ). Each retriever runs at `kEach = max(top_k * 3, 30)`; the
  merger trims to `top_k` for the caller.
- **Opt-in.** `[retrieval] retrieval_mode` selects `"dense"` (default) or
  `"hybrid"`. Validation rejects unknown values at config-load time.
  Reverting is one config line.

The change is transparent above the pipeline: the `Retrieve(ctx, query, k)`
contract is unchanged, the agent's `semantic_search` tool is unchanged, and
all callers are unchanged.

## Evidence

Full eval run on 85 questions, agent = local-llama / gemma4-e4b, judge =
remote-openai / gpt-5.4-mini, identical config except `retrieval_mode`.
Both result JSONs are archived under `eval/results/` for the memoria.

| Metric            | Baseline (dense) | Hybrid    | Δ |
|-------------------|------------------|-----------|---|
| **Overall mean**  | 76.1             | **80.2**  | **+4.1** |
| Faithfulness mean | 85.2             | 86.4      | +1.2 |
| Correctness mean  | 85.0             | 87.5      | +2.5 |
| Completeness mean | 81.6             | 83.7      | +2.1 |
| Recall@5          | 0.706            | 0.723     | +0.017 |
| Recall@10         | 0.786            | 0.790     | +0.004 |
| MRR               | 0.545            | 0.575     | +0.030 |
| Errors            | 0/85             | 0/85      | — |
| Cost (judge USD)  | $0.19            | $0.19     | — |

By category (Overall mean):

| Category   | n  | Baseline | Hybrid | Δ |
|------------|----|----------|--------|---|
| general    | 12 | 77.9     | **90.7** | **+12.8** |
| degree     | 24 | 82.3     | 85.5   | +3.2 |
| cross-doc  | 12 | 50.6     | 55.0   | +4.4 |
| course     | 12 | 74.7     | 76.9   | +2.2 |
| procedure  | 25 | 82.3     | 83.6   | +1.3 |

By difficulty (Overall mean):

| Diff   | n  | Baseline | Hybrid | Δ |
|--------|----|----------|--------|---|
| easy   | 44 | 77.3     | 83.5   | +6.2 |
| hard   | 12 | 50.6     | 55.0   | +4.4 |
| medium | 29 | 84.9     | 85.6   | +0.7 |

**Wins / losses:** 29 rows improved, 25 rows regressed, 31 unchanged.
Aggregate is positive but the change is non-uniform. The biggest wins are
in `general` — contact-info questions where exact tokens like
`secretariaetsii@uma.es`, phone numbers, and director names dominate the
BM25 signal. The biggest losses are queries where the corpus contains
multiple semantically-adjacent documents that share keywords (e.g. #61:
"Créditos obligatorios para defender TFM del MUII" — hybrid now also
surfaces the MUTDE master's calendar, and gemma4-e4b picks the wrong one,
mirroring the small-model distractor failure documented in ADR-007).

Recall@5 / @10 / MRR moved less than predicted (predicted +12–17 pts on
Recall@5; observed +1.7 pts). Most of the Overall lift comes from chunk
*selection within* the same top-K rather than from lifting previously
unseen golds. The agent now sees more keyword-rich chunks alongside the
semantic ones; better facts in the prompt → better answers, even when the
URL-level recall headline barely budges. Recall is a coarse measure for
this corpus because most rows have a single golden URL — Recall@5
collapses to a binary Hit@5 and is bounded by chunking rather than by
ranking once the canonical doc is the top hit.

## Consequences

- **Storage:** negligible. External-content FTS5 references the chunks
  table; the auxiliary FTS5 internal tables (`chunks_fts_data`,
  `chunks_fts_idx`, `chunks_fts_docsize`, `chunks_fts_config`) hold an
  inverted index whose size is sub-linear in chunk count. ≈10–20 MB on a
  corpus of low-thousands chunks.
- **Latency:** sub-millisecond per query. FTS5 is the same engine used by
  every iOS / Android / VS Code app shipping SQLite; well-trodden path.
- **No backfill.** The AFTER INSERT trigger indexes every chunk created
  post-migration; chunks that predate the `chunks_fts` table are not
  retroactively indexed. To populate the index over a pre-existing corpus,
  re-crawl the affected sources. Re-crawls are cheap and free, and skipping
  the conditional rebuild keeps the migration code surface minimal.
- **No new dependencies.** `modernc.org/sqlite` ships with FTS5 enabled;
  empirically verified during planning (`unicode61 remove_diacritics 2`
  and `bm25()` both behave correctly).
- **Concurrent-write test fixture updated.** The new triggers add work to
  every chunk INSERT, which exposed a SQLITE_BUSY in the cache
  thread-safety test (`TestConcurrent_SearchAndCreateChunks`). The test
  now serialises DB access via `db.SetMaxOpenConns(1)` so it focuses on
  the in-Go cache mutex (the actual unit under test). Production
  callers serialise `CreateChunks` per document anyway.
- **Follow-up: a reranker is the chosen next intervention.** RRF lifts
  the aggregate (+4.1 Overall) but isn't free: 25 of 85 rows regressed
  (post-hoc analysis of `hybrid_topk10_ctx4092.json` vs
  `baseline_topk10_ctx4092.json`, see ADR commit). The regressions cluster
  into 5 patterns — wrong-document substitution (~6 rows), refusal
  weakening on out-of-corpus questions (3 rows), mixed-doc hallucination
  (~3 rows), detail loss / hedging (~10 rows), and cross-doc rows that
  hybrid can't help structurally (4 rows). All five share a single root
  cause: **gemma4-e4b cannot reliably disambiguate lexically-adjacent
  chunks once they reach the prompt.** This is the same small-model
  distractor failure documented in ADR-007's wider-context A/B, just
  triggered by a different mechanism (better lexical recall instead of
  bigger context budget).

  The pattern across both ADRs makes the lever clear: small-model RAG is
  retrieval-precision-bound, not retrieval-coverage-bound. The next move
  must **suppress lexical near-misses before they reach the LLM**, not
  surface more candidates. A cross-encoder reranker over the top-30
  hybrid candidates is the canonical fix:

    - It re-scores each (query, chunk) pair jointly rather than via
      decomposable token overlap, so the MUTDE chunk that lexically
      matches an MUII question gets demoted because the cross-encoder
      sees the *whole* chunk doesn't actually answer the question.
    - It plugs in cleanly above the existing hybrid pipeline — same
      retrieve(top-30) call path, then a single rerank-and-truncate
      step before `BuildContext`.
    - The compute cost is bounded (top-30 cross-encoder calls ≈ a few
      hundred ms with a small Spanish-tuned encoder, well below the
      agent's per-turn budget) and additive — hybrid stays as the
      candidate-generation layer.

  Open design questions deferred to the implementation ADR (009):
  which encoder to use (Spanish-multilingual MS-MARCO variants vs.
  generic Qwen reranker), local vs remote hosting, latency budget per
  query, and whether to retrain on a slice of the eval dataset.

  **Deprioritised alternatives** — both still possible but smaller
  expected impact and less general:
    - FTS5 multi-column with weighted `bm25(table, 5.0, 1.0)` on
      `(section_title, content)` to lift canonical breadcrumbs. The
      current trigger already concatenates `coalesce(section_title, '')
      || ' ' || content` into a single FTS column (equal weighting); a
      multi-column variant would let heading words score higher than
      body, but adds tuning surface. Targeted; only fixes a subset of
      pattern 1 / 4.
    - Per-domain URL-canonicality boost preferring `/grado-en-X` over
      derivative `info/N/...` pages. Hand-coded heuristic; doesn't
      generalise to new corpora.

## Rejected alternatives

- **Wider context window** (ADR-007). Already rejected; -31 pts in A/B.
- **Weighted-score fusion** (`α · dense + (1-α) · BM25`). Requires
  per-corpus α tuning and score normalisation between two fundamentally
  different scales. RRF needs neither.
- **Pure BM25 replacement.** Loses dense's paraphrase strength on
  natural-language queries that don't share surface tokens with the
  source.
- **Per-section-title weighting.** Possible boost — give matches in
  `chunks.section_title` extra weight — but adds tuning surface.
  Re-evaluate after baseline hybrid lands.
- **Stemming or Spanish stop-word list.** Short factoid queries lose
  signal when stop-style tokens are dropped. Re-evaluate if eval shows
  systematic recall problems on long natural-language queries.
- **External BM25 (Bleve, Elastic, Tantivy via FFI).** New runtime
  dependencies, separate index lifecycle, no payoff over FTS5 at this
  corpus size.

## Critical files

- `internal/store/db.go` — FTS5 virtual table + triggers + backfill in
  `Migrate`.
- `internal/store/chunks.go` — `SearchKeyword`, `sanitizeFTS5Query`.
- `internal/rag/rrf.go` — `FuseRRF`.
- `internal/rag/retriever.go` — `Retrieve` mode switch, `buildRetrievalResult`.
- `internal/config/config.go` — `RetrievalConfig.RetrievalMode`, default,
  validation.
- `eval/scripts/d1_topk_probe.py` — diagnostic that motivated this work.
- `docs/decisions/007-rag-context-budget.md` — context for *why*
  ranking, not visibility, is the right lever.
