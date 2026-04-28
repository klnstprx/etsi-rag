# ADR-002: Brute-Force Cosine Similarity

## Status
Accepted — empirical numbers added 2026-04-30; the decision still holds.

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

| Path                                     | Latency  | Notes                                       |
|------------------------------------------|----------|---------------------------------------------|
| `embedder.Embed(query)` warm             | ~10 ms   | First call after server idle ~30–80 ms (one-shot llama-server first-inference setup, not TCP/connection); query length is not a factor |
| `SearchSimilar` topK=10, no filter       | ~368 ms  | 16,140 chunks, 498 MB allocated per call    |
| `SearchSimilar` topK=10, `language='es'` | ~394 ms  | join overhead; the filter is not selective at our corpus mix |
| `SearchSimilar` topK=50, no filter       | ~365 ms  | topK has negligible effect at this size     |
| In-memory cosine + top-K (preloaded)     | ~33 ms   | 192 KB allocated; isolates pure compute     |

**Where the time actually goes:** ~91% of `SearchSimilar` is spent loading 16K embedding BLOBs from SQLite and deserializing little-endian float32 (498 MB/op). Only ~9% is the cosine math. The synthetic table above modeled the compute path correctly — its 29 ms at 10K/dim=4096 is in the same order of magnitude as our 33 ms at 16K/dim=2560 for in-memory cosine — but underestimated the production path by ~11× because it omitted SQLite I/O + blob deserialization.

A typical retrieval tool call at the current corpus size is therefore **~378 ms** (10 ms embed + 368 ms search), not the ~30 ms the synthetic table suggested.

## Consequences
- Simple implementation, no extra dependencies — still holds.
- "Fast enough for ~5K–10K chunks" framing revised: 378 ms per retrieval tool call at 16K chunks is acceptable because the **agent loop is dominated by chat completions**, not retrieval. It is *not* fast in absolute terms.
- If retrieval latency becomes a UX concern, the highest-leverage fix is an **in-memory cache of deserialized embeddings** (would drop `SearchSimilar` from ~368 ms → ~33 ms with no algorithmic change). That is a much cheaper change than switching to sqlite-vec or HNSW, since the bottleneck is I/O, not the cosine math.
- If chunk count grows ~5× beyond 16K, revisit: the linear I/O cost will start to bite the user-perceived latency even with the LLM in the loop. Migrate to an in-memory cache first; only consider ANN once compute (not I/O) is the bottleneck.
- Latency scales linearly O(n) in both compute and I/O — predictable and easy to reason about.
