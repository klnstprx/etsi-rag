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
	mu           sync.RWMutex
	store        *store.SQLiteStore
	embedder     llm.Embedder
	chunker      chunking.Chunker
	counter      tokencount.Counter
	config       config.RetrievalConfig
	reranker     llm.Reranker
	rerankerPool int // candidate pool size for the rerank input; 0 = no over-fetch
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

// SetReranker swaps the reranker (thread-safe). Pass r=nil to disable rerank
// entirely, in which case poolSize is ignored. poolSize is the candidate-pool
// depth fed into the rerank input — Retrieve over-fetches up to this many
// candidates from each retriever (dense or hybrid) before reranking.
func (p *Pipeline) SetReranker(r llm.Reranker, poolSize int) {
	p.mu.Lock()
	p.reranker = r
	p.rerankerPool = poolSize
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

// getReranker returns the current reranker (may be nil) and its candidate-pool
// depth. Holding the read lock briefly is enough — callers should not retain
// the returned Reranker across config-rebuild boundaries.
func (p *Pipeline) getReranker() (llm.Reranker, int) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reranker, p.rerankerPool
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

// RetrieveOpts bundles the optional knobs accepted by Retrieve. The zero
// value selects the configured defaults and searches the full corpus.
//
// TopK overrides the configured top-k when non-nil.
//
// SourceKeys, when non-empty, restricts retrieval to chunks whose document is
// referenced from at least one crawl_pages row matching the given source key
// — the agent's list_sources / source_keys scoping path.
type RetrieveOpts struct {
	TopK       *int
	SourceKeys []string
}

// Retrieve finds relevant chunks for a query.
//
// In "dense" mode, scores chunks by cosine similarity over query embeddings
// (the long-standing default). In "hybrid" mode, combines dense and BM25
// keyword retrieval via reciprocal-rank fusion (see rrf.go and ADR-008).
//
// When a reranker is set on the pipeline (see SetReranker), Retrieve
// over-fetches a wider candidate pool, asks the reranker to re-score it, and
// trims the result to top-k. Reranker failures (network, timeout, tokenization
// trouble) fall through silently to the un-reranked ordering — chats must not
// fail because a reranker is offline. See ADR-009.
func (p *Pipeline) Retrieve(ctx context.Context, query string, opts RetrieveOpts) (*RetrievalResult, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	cfg := p.getConfig()

	k := cfg.TopK
	if opts.TopK != nil {
		k = *opts.TopK
	}
	sourceKeys := opts.SourceKeys

	reranker, rerankerPool := p.getReranker()

	var candidates []store.SearchResult
	switch cfg.RetrievalMode {
	case "", "dense":
		// When reranking is on, over-fetch so the cross-encoder has more than k
		// candidates to choose from. The candidate pool is at least the
		// reranker's configured TopN (typically 30).
		fetch := k
		if reranker != nil && rerankerPool > fetch {
			fetch = rerankerPool
		}
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		results, err := p.store.SearchSimilar(ctx, embedding, fetch, sourceKeys...)
		if err != nil {
			return nil, fmt.Errorf("searching: %w", err)
		}
		candidates = results

	case "hybrid":
		// Each leg gets a wider net; RRF picks the top-k of the union.
		// max(k*3, 30) is a defensible floor — the eval-D1 finding showed
		// most missed gold chunks rank at positions 7–12 in dense, so a
		// 30-deep candidate pool comfortably covers them. The reranker pool
		// raises the floor further when reranking is on.
		kEach := max(k*3, 30)
		if reranker != nil && rerankerPool > kEach {
			kEach = rerankerPool
		}
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		dense, err := p.store.SearchSimilar(ctx, embedding, kEach, sourceKeys...)
		if err != nil {
			return nil, fmt.Errorf("dense search: %w", err)
		}
		sparse, err := p.store.SearchKeyword(ctx, query, kEach, sourceKeys...)
		if err != nil {
			return nil, fmt.Errorf("keyword search: %w", err)
		}
		// Hybrid keeps a wider fused list when reranking; the reranker decides
		// what makes the cut. Without rerank, FuseRRF trims to k directly.
		fuseTo := k
		if reranker != nil && rerankerPool > fuseTo {
			fuseTo = rerankerPool
		}
		candidates = FuseRRF(dense, sparse, fuseTo, 60.0)

	default:
		return nil, fmt.Errorf("unsupported retrieval_mode: %q", cfg.RetrievalMode)
	}

	if reranker != nil && len(candidates) > 1 {
		candidates = applyRerank(ctx, reranker, query, candidates, k)
	}
	return buildRetrievalResult(candidates), nil
}

// RetrieveInDocument runs a dense cosine search restricted to a single
// document's chunks. Powers the read_document agent tool's query mode:
// when the LLM has a URL in hand and wants to drill into that doc with a
// follow-up question, this is the entry point.
//
// No hybrid / reranker variants in the first cut — within ≤50 candidates
// from one document the marginal precision wins from BM25 fusion or
// cross-encoder reranking aren't worth the complexity (and the recent
// pairwise eval showed those wins are inside judge noise even at full
// corpus scale). Revisit if read_document eval shows a precision gap.
func (p *Pipeline) RetrieveInDocument(ctx context.Context, query, docID string, topK int) (*RetrievalResult, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	embedding, err := embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	results, err := p.store.SearchInDocument(ctx, docID, embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("searching in document: %w", err)
	}
	return buildRetrievalResult(results), nil
}

// applyRerank re-scores candidates using the supplied reranker and trims to
// topK. On any failure (network, timeout, malformed response, partial result)
// the original ordering is returned trimmed to topK. We never bubble a
// reranker error up to the caller — chats must not fail because the
// reranker is offline.
func applyRerank(ctx context.Context, r llm.Reranker, query string, candidates []store.SearchResult, topK int) []store.SearchResult {
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		// Strip image markers — same hygiene the embed-time path uses
		// (Pipeline.Ingest), since they're metadata pointers, not content
		// the cross-encoder should score.
		docs[i] = markers.Strip(c.Chunk.Content)
	}
	results, err := r.Rerank(ctx, query, docs, topK)
	if err != nil || len(results) == 0 {
		return trimTo(candidates, topK)
	}
	out := make([]store.SearchResult, 0, len(results))
	for _, rr := range results {
		if rr.Index < 0 || rr.Index >= len(candidates) {
			continue
		}
		sr := candidates[rr.Index]
		// Replace with the cross-encoder's relevance score so SourceReference
		// surfaces it directly. The score is a raw logit (very small for
		// Qwen-style rerankers); ordering is the load-bearing property.
		sr.Score = rr.Score
		out = append(out, sr)
	}
	if len(out) == 0 {
		return trimTo(candidates, topK)
	}
	return trimTo(out, topK)
}

func trimTo(s []store.SearchResult, n int) []store.SearchResult {
	if n <= 0 || len(s) <= n {
		return s
	}
	return s[:n]
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
