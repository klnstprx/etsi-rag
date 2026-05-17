package llm

import "sync"

// EmbedderHandle is the single source of truth for the embedder shared between
// retrieval and ingestion. Both must use the same embedding model — stored
// chunk vectors live in one model's vector space, and a query embedded with a
// different model produces a vector in a different space, making cosine
// similarity meaningless. Routing the embedder through one handle makes that
// invariant unforgeable: a config reload swaps the embedder in one place and
// every consumer sees the new client on its next Get.
//
// A nil embedder is a valid state (resolve failure at startup or after a bad
// config save). Callers must nil-check Get().
type EmbedderHandle struct {
	mu sync.RWMutex
	e  Embedder
}

// NewEmbedderHandle constructs a handle with an initial embedder. Pass nil
// when the embedding backend is unconfigured or hasn't resolved yet.
func NewEmbedderHandle(e Embedder) *EmbedderHandle {
	return &EmbedderHandle{e: e}
}

// Set replaces the embedder. Pass nil to clear (used when a config reload
// fails to resolve a backend).
func (h *EmbedderHandle) Set(e Embedder) {
	h.mu.Lock()
	h.e = e
	h.mu.Unlock()
}

// Get returns the current embedder. May be nil; callers must handle that.
func (h *EmbedderHandle) Get() Embedder {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.e
}
