# ADR-002: Brute-Force Cosine Similarity

## Status
Accepted — empirical numbers added 2026-04-30; in-memory cache shipped 2026-04-30, dropping `SearchSimilar` from ~368 ms to ~37 ms at 16 K chunks.

## Context
Vector search is needed for RAG retrieval. Options: sqlite-vec extension (ANN index), external vector DB, or brute-force in Go.

## Decision
Brute-force cosine similarity computed in Go over `chunk_embeddings` BLOB column. No ANN index.

## Synthetic projections (initial)

AMD Ryzen 5 5600X, dim=4096, top-10, synthetic random float32 embeddings, **pure in-memory compute** (no SQLite I/O modeled):

| Chunks  | Latency | Verdict    |
|---------|---------|------------|
| 1,000   | 3.4 ms  | fine       |
| 5,000   | 15 ms   | acceptable |
| 10,000  | 29 ms   | noticeable |
| 50,000  | 149 ms  | sluggish   |
| 100,000 | 295 ms  | too slow   |

ANN index (simulated HNSW) would provide ~10–640× speedup at higher chunk counts.

## Empirical numbers (2026-04-30)

Real ETSI crawl, AMD Ryzen 5 5600X, dim=2,560 (Qwen3-Embedding-4B), embeddings served by llama-server with GPU offload. Reproducible via `internal/store/bench_test.go`:

```bash
go test -bench=BenchmarkSearchSimilarRealDB -benchmem -run='^$' -benchtime=5s ./internal/store/
```

| Path                                     | Latency  | Allocs/op  | Notes                                       |
|------------------------------------------|----------|------------|---------------------------------------------|
| `embedder.Embed(query)` warm             | ~10 ms   | —          | First call after server idle ~30–80 ms (one-shot llama-server first-inference setup, not TCP/connection); query length is not a factor |
| `SearchSimilar` topK=10, **uncached**    | ~372 ms  | 498 MB     | 16,140 chunks; full SQL scan + deserialize  |
| `SearchSimilar` topK=50, **uncached**    | ~371 ms  | 498 MB     | topK has negligible effect — load dominates |
| `SearchSimilar` topK=10, **cached**      | **~37 ms** | **307 KB** | post-Warm; metadata IN-fetch is the residual cost |
| In-memory cosine + top-K only            | ~33 ms   | 192 KB     | theoretical floor, no metadata fetch        |

**Where the time went pre-cache:** ~91% of `SearchSimilar` was spent loading 16 K embedding BLOBs from SQLite and deserializing little-endian float32 (498 MB/op). Only ~9% was the cosine math. The synthetic table above modeled the compute path correctly — 29 ms at 10 K/dim=4 096 is the same order of magnitude as 33 ms at 16 K/dim=2 560 for in-memory cosine — but underestimated the production path by ~11× because it omitted the SQLite I/O.

## In-memory cache (shipped 2026-04-30)

`SQLiteStore` now embeds an `embeddingCache` (a slice of `(chunkID, docID, vec)` guarded by `sync.RWMutex`). `Warm(ctx)` is called once at server startup and loads every embedding into memory; `CreateChunks`, `DeleteDocument`, and the purge methods write through. When the cache is unloaded (e.g. `:memory:` tests that never warm), `SearchSimilar` falls through to the legacy SQL scan.

Memory cost: `chunks × dim × 4 B`. At 16 K/dim=2 560: ~165 MB. At 100 K: ~1 GB. Linear, predictable.

## Consequences
- Simple implementation, no extra dependencies — still holds. The cache adds ~140 lines of Go in `internal/store/cache.go`; no new packages, no schema change, no ANN.
- A typical retrieval tool call at the current corpus size is now **~47 ms** (10 ms embed + 37 ms search), down from ~378 ms — the cache fully reclaims the I/O overhead that the original synthetic table had silently omitted.
- The remaining ~4 ms gap above the theoretical floor (33 ms) is the metadata `IN(...)` SQL roundtrip for the top-K rows. Cheap to optimize further by caching chunk content + document metadata too, but probably not worth the invalidation complexity at this scale.
- If chunk count grows ~5× beyond 16 K, the cache still scales linearly in compute (no I/O term) — at 100 K we project ~200 ms, still well under the LLM completion budget. Only consider ANN (sqlite-vec / HNSW) once *compute* (not I/O) is the bottleneck.
