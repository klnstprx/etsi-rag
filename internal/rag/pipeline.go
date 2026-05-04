// Package rag implements the Retrieval-Augmented Generation pipeline.
package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
)

// ErrChunking sentinels chunking-stage failures during Ingest so callers can
// distinguish them from embedding or storage failures (for circuit-breaker
// scoping, retry policy, etc.). Check with errors.Is.
var ErrChunking = errors.New("chunking failed")

// RetrievalResult represents chunks retrieved for a query.
type RetrievalResult struct {
	Chunks    []ScoredChunk
	Documents []domain.Document
}

// ScoredChunk is a chunk with its similarity score.
type ScoredChunk struct {
	Chunk domain.Chunk
	Score float32
}

// Pipeline orchestrates the RAG process: retrieval, context building, and ingestion.
type Pipeline struct {
	mu       sync.RWMutex
	store    *store.SQLiteStore
	embedder llm.Embedder
	chunker  chunking.Chunker
	counter  tokencount.Counter
	config   config.RetrievalConfig
}

// SetEmbedder swaps the embedding client (thread-safe).
func (p *Pipeline) SetEmbedder(e llm.Embedder) {
	p.mu.Lock()
	p.embedder = e
	p.mu.Unlock()
}

// SetChunker swaps the chunker (thread-safe).
func (p *Pipeline) SetChunker(ch chunking.Chunker) {
	p.mu.Lock()
	p.chunker = ch
	p.mu.Unlock()
}

// SetCounter swaps the token counter (thread-safe).
func (p *Pipeline) SetCounter(c tokencount.Counter) {
	p.mu.Lock()
	p.counter = c
	p.mu.Unlock()
}

// SetConfig swaps the RAG config (thread-safe).
func (p *Pipeline) SetConfig(cfg config.RetrievalConfig) {
	p.mu.Lock()
	p.config = cfg
	p.mu.Unlock()
}

func (p *Pipeline) getEmbedder() llm.Embedder {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.embedder
}

func (p *Pipeline) getChunker() chunking.Chunker {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.chunker
}

func (p *Pipeline) getCounter() tokencount.Counter {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.counter
}

func (p *Pipeline) getConfig() config.RetrievalConfig {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.config
}

// NewPipeline creates a new RAG pipeline.
// Counter may be nil at startup (misconfigured backend); operations requiring
// token counting will return errors until a valid counter is set via SetCounter.
func NewPipeline(s *store.SQLiteStore, embedder llm.Embedder, ch chunking.Chunker, counter tokencount.Counter, cfg config.RetrievalConfig) *Pipeline {
	return &Pipeline{
		store:    s,
		embedder: embedder,
		chunker:  ch,
		counter:  counter,
		config:   cfg,
	}
}

// Preflight checks that the embedding backend is configured and reachable.
func (p *Pipeline) Preflight(ctx context.Context) error {
	embedder := p.getEmbedder()
	if embedder == nil {
		return fmt.Errorf("embedding backend not configured")
	}
	_, err := embedder.Embed(ctx, "ping")
	if err != nil {
		return fmt.Errorf("embedding backend unreachable: %w", err)
	}
	return nil
}

// Retrieve finds relevant chunks for a query.
//
// In "dense" mode, scores chunks by cosine similarity over query embeddings
// (the long-standing default). In "hybrid" mode, combines dense and BM25
// keyword retrieval via reciprocal-rank fusion (see rrf.go and ADR-008).
func (p *Pipeline) Retrieve(ctx context.Context, query string, topK *int) (*RetrievalResult, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	cfg := p.getConfig()

	k := cfg.TopK
	if topK != nil {
		k = *topK
	}

	switch cfg.RetrievalMode {
	case "", "dense":
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		results, err := p.store.SearchSimilar(ctx, embedding, k)
		if err != nil {
			return nil, fmt.Errorf("searching: %w", err)
		}
		return buildRetrievalResult(results), nil

	case "hybrid":
		// Each leg gets a wider net; RRF picks the top-k of the union.
		// max(k*3, 30) is a defensible floor — the eval-D1 finding showed
		// most missed gold chunks rank at positions 7–12 in dense, so a
		// 30-deep candidate pool comfortably covers them.
		kEach := max(k*3, 30)
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		dense, err := p.store.SearchSimilar(ctx, embedding, kEach)
		if err != nil {
			return nil, fmt.Errorf("dense search: %w", err)
		}
		sparse, err := p.store.SearchKeyword(ctx, query, kEach)
		if err != nil {
			return nil, fmt.Errorf("keyword search: %w", err)
		}
		merged := FuseRRF(dense, sparse, k, 60.0)
		return buildRetrievalResult(merged), nil

	default:
		return nil, fmt.Errorf("unsupported retrieval_mode: %q", cfg.RetrievalMode)
	}
}

// buildRetrievalResult collapses a list of SearchResults into a
// RetrievalResult, deduping documents while preserving chunk order.
func buildRetrievalResult(results []store.SearchResult) *RetrievalResult {
	chunks := make([]ScoredChunk, 0, len(results))
	seen := map[string]bool{}
	var docs []domain.Document

	for _, r := range results {
		chunks = append(chunks, ScoredChunk{
			Chunk: r.Chunk,
			Score: r.Score,
		})
		if !seen[r.Document.ID] {
			seen[r.Document.ID] = true
			docs = append(docs, r.Document)
		}
	}
	return &RetrievalResult{Chunks: chunks, Documents: docs}
}

// BuildContext constructs the context string from retrieval results.
func (p *Pipeline) BuildContext(ctx context.Context, results *RetrievalResult) (string, error) {
	if results == nil || len(results.Chunks) == 0 {
		return "", nil
	}

	counter := p.getCounter()
	if counter == nil {
		return "", fmt.Errorf("token counter not configured")
	}
	cfg := p.getConfig()

	// Map document IDs to titles
	docTitles := make(map[string]string)
	for _, d := range results.Documents {
		docTitles[d.ID] = d.Title
	}

	var b strings.Builder
	currentTokens := 0

	for i, sc := range results.Chunks {
		title := docTitles[sc.Chunk.DocID]
		header := title
		if sc.Chunk.SectionTitle != "" {
			header = title + " > " + sc.Chunk.SectionTitle
		}
		block := fmt.Sprintf("[%d] %s:\n%s\n\n", i+1, header, sc.Chunk.Content)

		blockTokens, err := counter.Count(ctx, block)
		if err != nil {
			return "", fmt.Errorf("counting context tokens: %w", err)
		}

		if currentTokens+blockTokens > cfg.MaxContextTokens {
			break
		}
		b.WriteString(block)
		currentTokens += blockTokens
	}

	return b.String(), nil
}

// SourcesFromRetrieval builds source references from retrieval results.
func SourcesFromRetrieval(results *RetrievalResult) []domain.SourceReference {
	if results == nil {
		return nil
	}
	docMap := make(map[string]domain.Document)
	for _, d := range results.Documents {
		docMap[d.ID] = d
	}
	var sources []domain.SourceReference
	for _, sc := range results.Chunks {
		doc := docMap[sc.Chunk.DocID]
		sources = append(sources, domain.SourceReference{
			ID:         doc.ID,
			Title:      doc.Title,
			URL:        doc.URL,
			ChunkIndex: sc.Chunk.ChunkIndex,
			Score:      sc.Score,
		})
	}
	return sources
}

// Ingest adds documents to the RAG index.
// Returns (documents ingested, chunks created, error).
func (p *Pipeline) Ingest(ctx context.Context, docs []domain.Document) (int, int, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return 0, 0, fmt.Errorf("embedding backend not configured")
	}

	var totalDocs, totalChunks int

	for _, doc := range docs {
		if err := p.store.CreateDocument(ctx, &doc); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing document %s: %w", doc.ID, err)
		}

		chunks, err := p.getChunker().Chunk(ctx, doc.Content)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("chunking document %s: %w: %w", doc.ID, ErrChunking, err)
		}
		if len(chunks) == 0 {
			totalDocs++
			continue
		}

		// Convert chunking.Chunk to domain.Chunk and collect texts. Strip image
		// markers from the embed-time text so they don't pollute the embedding
		// space; the storage path keeps the markers so retrieval can resolve
		// them to image artifacts.
		domainChunks := make([]domain.Chunk, len(chunks))
		texts := make([]string, len(chunks))
		for i, c := range chunks {
			domainChunks[i] = domain.Chunk{
				Content:      c.Content,
				ChunkIndex:   c.Index,
				SectionTitle: c.SectionTitle,
			}
			texts[i] = markers.Strip(c.Content)
		}

		embeddings, err := embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("embedding chunks for %s: %w", doc.ID, err)
		}

		if err := p.store.CreateChunks(ctx, doc.ID, domainChunks, embeddings); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing chunks for %s: %w", doc.ID, err)
		}

		totalDocs++
		totalChunks += len(chunks)
	}

	return totalDocs, totalChunks, nil
}
