// Package ingest implements the indexing side of the corpus: chunking
// documents, embedding the resulting chunks, and persisting them to the
// store. It is the write counterpart of internal/rag (which is read-only
// retrieval). Crawler runs and the REST ingest endpoint both go through
// here; query-time consumers depend on *rag.Retriever instead.
package ingest

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
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

	titleInSearch bool // see config.ChunkingConfig.TitleInSearchText
}

// New wires the dependencies. The embedder handle may already hold
// nil at startup (misconfigured backend); Ingest and HealthCheck both surface
// "embedding backend not configured" until a real client is set.
func New(s *store.SQLiteStore, embedder *llm.EmbedderHandle, ch chunking.Chunker) *Ingester {
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

func (i *Ingester) currentChunker() chunking.Chunker {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.chunker
}

// SetTitleInSearch sets whether chunks are embedded and keyword-indexed with
// their document title ahead of the breadcrumb (thread-safe).
func (i *Ingester) SetTitleInSearch(on bool) {
	i.mu.Lock()
	i.titleInSearch = on
	i.mu.Unlock()
}

// searchPrefix is the text put ahead of every chunk breadcrumb of doc.
func (i *Ingester) searchPrefix(doc domain.Document) string {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if !i.titleInSearch {
		return ""
	}
	return doc.Title
}

// HealthCheck verifies the embedding backend is configured and reachable by
// issuing a one-token "ping" embed, and that the chunker's tokenizer answers.
// Used by the crawler and admin handler to fail fast at the start of a
// crawl/reprocess run, before workers spawn and partial side effects accumulate.
func (i *Ingester) HealthCheck(ctx context.Context) error {
	embedder := i.embedder.Get()
	if embedder == nil {
		return fmt.Errorf("embedding backend not configured")
	}
	if _, err := embedder.Embed(ctx, "ping"); err != nil {
		return fmt.Errorf("embedding backend unreachable: %w", err)
	}
	if _, err := i.currentChunker().Chunk(ctx, "ping"); err != nil {
		return fmt.Errorf("chunker unavailable: %w", err)
	}
	return nil
}

// Ingest writes documents into the corpus. Returns (documents ingested,
// chunks created, error). The documents row is written after its chunks are
// embedded and removed again if storing them fails, so a failure never leaves
// a document without chunks.
//
// Chunking failures are wrapped with ErrChunking so callers can distinguish
// them from embedding/storage failures — the crawler's consecutive-embed-fail
// circuit breaker uses this to avoid tripping on document-specific chunking
// quirks (the backend is healthy; one document just can't be processed). A
// tokenizer backend that is down (tokencount.ErrBackend) is not one of them.
func (i *Ingester) Ingest(ctx context.Context, docs []domain.Document) (int, int, error) {
	embedder := i.embedder.Get()
	if embedder == nil {
		return 0, 0, fmt.Errorf("embedding backend not configured")
	}

	var totalDocs, totalChunks int

	for _, doc := range docs {
		chunks, err := i.currentChunker().Chunk(ctx, doc.Content)
		if errors.Is(err, tokencount.ErrBackend) {
			return totalDocs, totalChunks, fmt.Errorf("chunking document %s: %w", doc.ID, err)
		}
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("chunking document %s: %w: %w", doc.ID, ErrChunking, err)
		}

		// Convert chunking.Chunk to domain.Chunk. Image refs travel as a
		// separate field on the chunk record; chunk content is already
		// marker-free at this point (the chunker strips markers during
		// emission). The breadcrumb lives only on SectionTitle — the
		// embedder sees it via chunking.EmbeddingText, but stored content
		// is plain body so the LLM-facing context block (which already
		// renders DocTitle > SectionTitle in its header) does not
		// double-render the path.
		prefix := i.searchPrefix(doc)
		domainChunks := make([]domain.Chunk, len(chunks))
		texts := make([]string, len(chunks))
		for j, c := range chunks {
			domainChunks[j] = domain.Chunk{
				Content:      c.Content,
				ChunkIndex:   c.Index,
				SectionTitle: c.SectionTitle,
				ImageRefs:    c.ImageRefs,
				SearchPrefix: prefix,
			}
			texts[j] = chunking.EmbeddingText(chunking.SearchBreadcrumb(prefix, c.SectionTitle), c.Content)
		}

		var embeddings [][]float32
		if len(texts) > 0 {
			embeddings, err = embedder.EmbedBatch(ctx, texts)
			if err != nil {
				return totalDocs, totalChunks, fmt.Errorf("embedding chunks for %s: %w", doc.ID, err)
			}
		}

		if err := i.store.CreateDocument(ctx, &doc); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing document %s: %w", doc.ID, err)
		}
		if err := i.store.CreateChunks(ctx, doc.ID, domainChunks, embeddings); err != nil {
			// Later pages with the same content would adopt a chunkless
			// document, so the row goes too, even when ctx is cancelled.
			if delErr := i.store.DeleteDocument(context.WithoutCancel(ctx), doc.ID); delErr != nil {
				err = errors.Join(err, delErr)
			}
			return totalDocs, totalChunks, fmt.Errorf("storing chunks for %s: %w", doc.ID, err)
		}

		totalDocs++
		totalChunks += len(chunks)
	}

	return totalDocs, totalChunks, nil
}
