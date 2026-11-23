package store

import (
	"fmt"
	"sync"
)

// cachedVec is one entry in the in-memory embedding cache. The vector is the
// already-deserialized []float32, so SearchSimilar avoids the SQLite BLOB
// load + little-endian decode that dominated its latency in ADR-002.
type cachedVec struct {
	ChunkID int64
	DocID   string
	Vec     []float32
}

// scoredChunk is a chunk ID with its retrieval score, higher is better. Every
// search ranks these first and fetches chunk and document metadata only for
// the winners.
type scoredChunk struct {
	ChunkID int64
	Score   float32
}

// embeddingCache is an in-memory mirror of chunk_embeddings keyed by chunkID,
// guarded by a single RWMutex. Mirrors the pattern in internal/session/store.go
// (read-heavy, infrequent writes).
//
// The zero value is an unloaded cache: searchTopK signals the caller to fall
// back to the SQL path, and append/remove are no-ops. This keeps :memory:
// tests that never call Warm working unchanged.
type embeddingCache struct {
	mu     sync.RWMutex
	vecs   []cachedVec
	dim    int
	loaded bool
}

// warm replaces cache contents with vecs and marks the cache loaded. Called
// from SQLiteStore.Warm after a single JOIN scan of the DB.
func (c *embeddingCache) warm(vecs []cachedVec, dim int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.vecs = vecs
	c.dim = dim
	c.loaded = true
}

// appendChunks adds entries after a successful CreateChunks commit. No-op when
// the cache is unloaded (the SQL fallback covers it; a later Warm rebuilds
// from authoritative DB state).
//
// A dim mismatch is treated as a signal that the embedding model changed under
// us — typically the reprocess-after-model-change path. The DB row was already
// committed at the new dim, so the cache invalidates itself (drop entries,
// mark unloaded) and lets SearchSimilar fall back to SQL until a later Warm
// rebuilds the cache. Returning an error here would fail every reprocessed
// page, which is exactly what the operator runs reprocess to fix.
func (c *embeddingCache) appendChunks(entries []cachedVec) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.loaded {
		return
	}
	for _, e := range entries {
		if len(e.Vec) != c.dim {
			c.vecs = nil
			c.dim = 0
			c.loaded = false
			return
		}
	}
	c.vecs = append(c.vecs, entries...)
}

// removeByDocIDs drops all cached entries whose DocID is in docIDs. Used by
// DeleteDocument and the purge methods after their DB DELETE commits.
func (c *embeddingCache) removeByDocIDs(docIDs []string) {
	if len(docIDs) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.loaded {
		return
	}
	doomed := make(map[string]struct{}, len(docIDs))
	for _, id := range docIDs {
		doomed[id] = struct{}{}
	}
	kept := c.vecs[:0]
	for _, v := range c.vecs {
		if _, drop := doomed[v.DocID]; drop {
			continue
		}
		kept = append(kept, v)
	}
	for i := len(kept); i < len(c.vecs); i++ {
		c.vecs[i] = cachedVec{}
	}
	c.vecs = kept
}

// searchTopK scores query against every cached vector and returns the top-K
// entries by cosine similarity descending.
//
// Returns useCache=false when the cache is unloaded — the caller should run
// the legacy SQL search path. Returns an empty slice with useCache=true when
// the cache is loaded but empty.
func (c *embeddingCache) searchTopK(query []float32, topK int) (results []scoredChunk, useCache bool, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.loaded {
		return nil, false, nil
	}
	if c.dim != 0 && len(query) != c.dim {
		return nil, true, fmt.Errorf("query dim %d != cache dim %d", len(query), c.dim)
	}

	scored := make([]scoredChunk, 0, len(c.vecs))
	for _, v := range c.vecs {
		scored = append(scored, scoredChunk{ChunkID: v.ChunkID, Score: cosineSimilarity(query, v.Vec)})
	}
	if len(scored) == 0 {
		return nil, true, nil
	}
	return rankTopK(scored, topK), true, nil
}
