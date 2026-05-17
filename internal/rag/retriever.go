// Package rag implements the retrieval side of the Retrieval-Augmented
// Generation system: dense and hybrid search, optional cross-encoder
// reranking, and context-block construction for LLM consumption. Ingestion
// (chunking + embedding + storage) is the responsibility of internal/ingest.
package rag

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
)

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

// RetrieveOpts bundles the optional knobs accepted by Retrieve. The zero
// value selects the configured defaults and searches the full corpus.
//
// TopK overrides the configured top-k when non-nil.
type RetrieveOpts struct {
	TopK *int
}

// Retriever serves query-side reads: it embeds the query, searches the
// store, optionally fuses dense+BM25 and runs a reranker, and formats the
// result into an LLM-ready context block.
//
// The embedder is shared with internal/ingest via an EmbedderHandle so the
// invariant "query embeddings and chunk embeddings live in the same vector
// space" is enforced by construction (one handle, one swap point on config
// reload).
type Retriever struct {
	mu           sync.RWMutex
	store        *store.SQLiteStore
	embedder     *llm.EmbedderHandle
	counter      tokencount.Counter
	config       config.RetrievalConfig
	reranker     llm.Reranker
	rerankerPool int // candidate pool size for the rerank input; 0 = no over-fetch
}

// NewRetriever creates a Retriever wired against the given dependencies.
// Counter may be nil at startup (misconfigured backend); BuildContext will
// return an error until a valid counter is set via SetCounter.
func NewRetriever(s *store.SQLiteStore, embedder *llm.EmbedderHandle, counter tokencount.Counter, cfg config.RetrievalConfig) *Retriever {
	return &Retriever{
		store:    s,
		embedder: embedder,
		counter:  counter,
		config:   cfg,
	}
}

// SetCounter swaps the token counter (thread-safe).
func (r *Retriever) SetCounter(c tokencount.Counter) {
	r.mu.Lock()
	r.counter = c
	r.mu.Unlock()
}

// SetConfig swaps the retrieval config (thread-safe).
func (r *Retriever) SetConfig(cfg config.RetrievalConfig) {
	r.mu.Lock()
	r.config = cfg
	r.mu.Unlock()
}

// SetReranker swaps the reranker (thread-safe). Pass r=nil to disable rerank
// entirely, in which case poolSize is ignored. poolSize is the candidate-pool
// depth fed into the rerank input — Retrieve over-fetches up to this many
// candidates from each retriever (dense or hybrid) before reranking.
func (r *Retriever) SetReranker(re llm.Reranker, poolSize int) {
	r.mu.Lock()
	r.reranker = re
	r.rerankerPool = poolSize
	r.mu.Unlock()
}

func (r *Retriever) getCounter() tokencount.Counter {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.counter
}

func (r *Retriever) getConfig() config.RetrievalConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.config
}

// getReranker returns the current reranker (may be nil) and its candidate-pool
// depth. Holding the read lock briefly is enough — callers should not retain
// the returned Reranker across config-rebuild boundaries.
func (r *Retriever) getReranker() (llm.Reranker, int) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.reranker, r.rerankerPool
}

// Retrieve finds relevant chunks for a query.
//
// In "dense" mode, scores chunks by cosine similarity over query embeddings
// (the long-standing default). In "hybrid" mode, combines dense and BM25
// keyword retrieval via reciprocal-rank fusion (see rrf.go).
//
// When a reranker is set on the retriever (see SetReranker), Retrieve
// over-fetches a wider candidate pool, asks the reranker to re-score it, and
// trims the result to top-k. Reranker failures (network, timeout, tokenization
// trouble) fall through silently to the un-reranked ordering — chats must not
// fail because a reranker is offline.
func (r *Retriever) Retrieve(ctx context.Context, query string, opts RetrieveOpts) (*RetrievalResult, error) {
	embedder := r.embedder.Get()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	cfg := r.getConfig()

	k := cfg.TopK
	if opts.TopK != nil {
		k = *opts.TopK
	}

	reranker, rerankerPool := r.getReranker()

	var candidates []store.SearchResult
	switch cfg.RetrievalMode {
	case "", "dense":
		fetch := k
		if reranker != nil && rerankerPool > fetch {
			fetch = rerankerPool
		}
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		results, err := r.store.SearchSimilar(ctx, embedding, fetch)
		if err != nil {
			return nil, fmt.Errorf("searching: %w", err)
		}
		candidates = results

	case "hybrid":
		kEach := max(k*3, 30)
		if reranker != nil && rerankerPool > kEach {
			kEach = rerankerPool
		}
		embedding, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		dense, err := r.store.SearchSimilar(ctx, embedding, kEach)
		if err != nil {
			return nil, fmt.Errorf("dense search: %w", err)
		}
		sparse, err := r.store.SearchKeyword(ctx, query, kEach)
		if err != nil {
			return nil, fmt.Errorf("keyword search: %w", err)
		}
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
// cross-encoder reranking aren't worth the complexity.
func (r *Retriever) RetrieveInDocument(ctx context.Context, query, docID string, topK int) (*RetrievalResult, error) {
	embedder := r.embedder.Get()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	embedding, err := embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	results, err := r.store.SearchInDocument(ctx, docID, embedding, topK)
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
func applyRerank(ctx context.Context, re llm.Reranker, query string, candidates []store.SearchResult, topK int) []store.SearchResult {
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		docs[i] = c.Chunk.Content
	}
	results, err := re.Rerank(ctx, query, docs, topK)
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

// BuildContext constructs the context string from retrieval results, fitting
// as many chunks as the configured MaxContextTokens budget allows.
func (r *Retriever) BuildContext(ctx context.Context, results *RetrievalResult) (string, error) {
	if results == nil || len(results.Chunks) == 0 {
		return "", nil
	}

	counter := r.getCounter()
	if counter == nil {
		return "", fmt.Errorf("token counter not configured")
	}
	cfg := r.getConfig()

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
