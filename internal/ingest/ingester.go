// Package ingest implements the indexing side of the corpus: chunking
// documents, embedding the resulting chunks, and persisting them to the
// store. It is the write counterpart of internal/rag (which is read-only
// retrieval). Crawler runs and the REST ingest endpoint both go through
// here; query-time consumers depend on *rag.Retriever instead.
package ingest

import (
	"context"
	"fmt"
	"sync"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
)

// Ingester writes documents into the corpus: chunk → embed → store.
//
// The embedder is shared with retrieval via an EmbedderHandle so the
// invariant "the model that embedded a chunk must also embed queries against
// it" is enforced by construction (single source of truth, swapped in one
// place on config reload). The chunker is per-Ingester because it's the only
// thing the ingest path tunes independently.
type Ingester struct {
	mu       sync.RWMutex
	store    *store.SQLiteStore
	embedder *llm.EmbedderHandle
	chunker  chunking.Chunker
}

// NewIngester wires the dependencies. The embedder handle may already hold
// nil at startup (misconfigured backend); Ingest and HealthCheck both surface
// "embedding backend not configured" until a real client is set.
func NewIngester(s *store.SQLiteStore, embedder *llm.EmbedderHandle, ch chunking.Chunker) *Ingester {
	return &Ingester{
		store:    s,
		embedder: embedder,
		chunker:  ch,
	}
}

// SetChunker swaps the chunker (thread-safe). Called from server.rebuildClients
// when a config save changes chunking parameters or the embedding model
// (which changes the tokenizer the chunker uses internally).
func (i *Ingester) SetChunker(ch chunking.Chunker) {
	i.mu.Lock()
	i.chunker = ch
	i.mu.Unlock()
}

func (i *Ingester) getChunker() chunking.Chunker {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.chunker
}

// HealthCheck verifies the embedding backend is configured and reachable by
// issuing a one-token "ping" embed. Used by the crawler and admin handler
// to fail fast at the start of a crawl/reprocess run, before workers spawn
// and partial side effects accumulate.
func (i *Ingester) HealthCheck(ctx context.Context) error {
	embedder := i.embedder.Get()
	if embedder == nil {
		return fmt.Errorf("embedding backend not configured")
	}
	if _, err := embedder.Embed(ctx, "ping"); err != nil {
		return fmt.Errorf("embedding backend unreachable: %w", err)
	}
	return nil
}

// Ingest writes documents into the corpus. Returns (documents ingested,
// chunks created, error).
//
// Chunking failures are wrapped with ErrChunking so callers can distinguish
// them from embedding/storage failures — the crawler's consecutive-embed-fail
// circuit breaker uses this to avoid tripping on document-specific chunking
// quirks (the backend is healthy; one document just can't be processed).
func (i *Ingester) Ingest(ctx context.Context, docs []domain.Document) (int, int, error) {
	embedder := i.embedder.Get()
	if embedder == nil {
		return 0, 0, fmt.Errorf("embedding backend not configured")
	}

	var totalDocs, totalChunks int

	for _, doc := range docs {
		if err := i.store.CreateDocument(ctx, &doc); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing document %s: %w", doc.ID, err)
		}

		chunks, err := i.getChunker().Chunk(ctx, doc.Content)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("chunking document %s: %w: %w", doc.ID, ErrChunking, err)
		}
		if len(chunks) == 0 {
			totalDocs++
			continue
		}

		// Convert chunking.Chunk to domain.Chunk. Image refs travel as a
		// separate field on the chunk record; chunk content is already
		// marker-free at this point (the chunker strips markers during
		// emission). The breadcrumb lives only on SectionTitle — the
		// embedder sees it via chunking.EmbeddingText, but stored content
		// is plain body so the LLM-facing context block (which already
		// renders DocTitle > SectionTitle in its header) does not
		// double-render the path.
		domainChunks := make([]domain.Chunk, len(chunks))
		texts := make([]string, len(chunks))
		for j, c := range chunks {
			domainChunks[j] = domain.Chunk{
				Content:      c.Content,
				ChunkIndex:   c.Index,
				SectionTitle: c.SectionTitle,
				ImageRefs:    c.ImageRefs,
			}
			texts[j] = chunking.EmbeddingText(c.SectionTitle, c.Content)
		}

		embeddings, err := embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("embedding chunks for %s: %w", doc.ID, err)
		}

		if err := i.store.CreateChunks(ctx, doc.ID, domainChunks, embeddings); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing chunks for %s: %w", doc.ID, err)
		}

		totalDocs++
		totalChunks += len(chunks)
	}

	return totalDocs, totalChunks, nil
}
