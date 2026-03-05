# ADR-002: Brute-Force Cosine Similarity

## Status
Accepted

## Context
Vector search is needed for RAG retrieval. Options: sqlite-vec extension (ANN index), external vector DB, or brute-force in Go.

## Decision
Brute-force cosine similarity computed in Go over `chunk_embeddings` BLOB column. No ANN index.

## Benchmarks

AMD Ryzen 5 5600X, dim=4096, top-10, synthetic random float32 embeddings:

| Chunks  | Latency | Verdict    |
|---------|---------|------------|
| 1,000   | 3.4ms   | fine       |
| 5,000   | 15ms    | acceptable |
| 10,000  | 29ms    | noticeable |
| 50,000  | 149ms   | sluggish   |
| 100,000 | 295ms   | too slow   |

ANN index (simulated HNSW) would provide ~10-640x speedup at higher chunk counts.

## Consequences
- Simple implementation, no extra dependencies
- Fast enough for expected dataset (~5K-10K chunks from ETSI crawl)
- If dataset grows beyond ~10K chunks, migrate to sqlite-vec or similar ANN index
- Latency scales linearly O(n) — predictable and easy to reason about
