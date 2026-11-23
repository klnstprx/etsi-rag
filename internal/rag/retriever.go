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

	"github.com/rs/zerolog"

	"etsi-rag/internal/chunking"
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
	mu       sync.RWMutex
	store    *store.SQLiteStore
	embedder *llm.EmbedderHandle
	counter  tokencount.Counter
	config   config.RetrievalConfig
	rerank   RerankerSetup
}

// RerankerSetup configures the rerank step of Retrieve. A nil Reranker
// disables it, and the other fields are then ignored.
type RerankerSetup struct {
	Reranker llm.Reranker
	// Pool is the candidate-pool depth fed into the rerank input.
	Pool int
	// TitleInText sends each candidate as "title > breadcrumb:\nbody" instead
	// of the body alone; see config.RerankerConfig.TitleInText.
	TitleInText bool
	Logger      zerolog.Logger
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

// SetReranker swaps the rerank step (thread-safe). Retrieve over-fetches up to
// setup.Pool candidates from each retriever (dense or hybrid) before reranking.
func (r *Retriever) SetReranker(setup RerankerSetup) {
	r.mu.Lock()
	r.rerank = setup
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

// getReranker returns the current rerank setup. Holding the read lock briefly
// is enough — callers should not retain it across config-rebuild boundaries.
func (r *Retriever) getReranker() RerankerSetup {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.rerank
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
// trouble) are logged and fall through to the un-reranked ordering — chats
// must not fail because a reranker is offline.
func (r *Retriever) Retrieve(ctx context.Context, query string, opts RetrieveOpts) (*RetrievalResult, error) {
	embedder := r.embedder.Get()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}
	cfg := r.getConfig()
	hybrid := cfg.RetrievalMode == "hybrid"
	if !hybrid && cfg.RetrievalMode != "" && cfg.RetrievalMode != "dense" {
		return nil, fmt.Errorf("unsupported retrieval_mode: %q", cfg.RetrievalMode)
	}

	k := cfg.TopK
	if opts.TopK != nil {
		k = *opts.TopK
	}

	// pool is how many candidates reach the reranker (or the result, without one).
	rerank := r.getReranker()
	pool := k
	if rerank.Reranker != nil {
		pool = max(k, rerank.Pool)
	}

	embedding, err := embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	var candidates []store.SearchResult
	if hybrid {
		// Each list over-fetches so fusion has overlap to reward.
		perList := max(k*3, 30, pool)
		dense, err := r.store.SearchSimilar(ctx, embedding, perList)
		if err != nil {
			return nil, fmt.Errorf("dense search: %w", err)
		}
		sparse, err := r.store.SearchKeyword(ctx, query, perList)
		if err != nil {
			return nil, fmt.Errorf("keyword search: %w", err)
		}
		candidates = FuseRRF(dense, sparse, pool, rrfK)
	} else {
		candidates, err = r.store.SearchSimilar(ctx, embedding, pool)
		if err != nil {
			return nil, fmt.Errorf("searching: %w", err)
		}
	}

	if rerank.Reranker != nil && len(candidates) > 1 {
		candidates = applyRerank(ctx, rerank, query, candidates, k)
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
func applyRerank(ctx context.Context, rerank RerankerSetup, query string, candidates []store.SearchResult, topK int) []store.SearchResult {
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		docs[i] = c.Chunk.Content
		if rerank.TitleInText {
			docs[i] = chunking.EmbeddingText(chunking.SearchBreadcrumb(c.Document.Title, c.Chunk.SectionTitle), c.Chunk.Content)
		}
	}
	results, err := rerank.Reranker.Rerank(ctx, query, docs, topK)
	if err != nil {
		rerank.Logger.Warn().Err(err).Int("candidates", len(docs)).Msg("rerank failed; keeping the retrieval order")
		return trimTo(candidates, topK)
	}
	if len(results) == 0 {
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

	docs := make(map[string]domain.Document, len(results.Documents))
	for _, d := range results.Documents {
		docs[d.ID] = d
	}

	var b strings.Builder
	currentTokens := 0

	for i, sc := range results.Chunks {
		block := ContextBlock(i+1, docs[sc.Chunk.DocID], sc.Chunk)

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

// ContextBlock renders one retrieved chunk as the LLM sees it: its 1-based
// rank, the document title with the chunk's section as a breadcrumb, the
// document URL (the argument read_document takes) when known, and the chunk
// text. Both agent tools use it so the model sees one shape.
func ContextBlock(rank int, doc domain.Document, c domain.Chunk) string {
	header := doc.Title
	if c.SectionTitle != "" {
		header = doc.Title + " > " + c.SectionTitle
	}
	if doc.URL != nil && *doc.URL != "" {
		return fmt.Sprintf("[%d] %s:\nURL: %s\n%s\n\n", rank, header, *doc.URL, c.Content)
	}
	return fmt.Sprintf("[%d] %s:\n%s\n\n", rank, header, c.Content)
}
