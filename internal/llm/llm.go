// Package llm provides the LLM client using the official openai-go SDK.
// llama-server is OpenAI API compliant, so we use the SDK directly.
package llm

import "context"

// Embedder defines the interface for embedding operations.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// RerankResult is one (document, score) pair returned by a reranker. Index
// refers back to the position of the document in the request's input list, so
// callers can map results to their original candidate set without sending
// document identifiers over the wire.
//
// Score is the raw relevance score returned by the backend. For llama-server
// /rerank with a Qwen-style cross-encoder this is a pre-softmax logit (very
// small magnitude, e.g. 1e-22). The relative ordering is what matters; the
// absolute value is preserved here for debugging / observability.
type RerankResult struct {
	Index int
	Score float32
}

// Reranker re-scores a list of candidate documents against a query and returns
// them in descending relevance order.
//
// topN is the maximum number of results the caller wants back; backends may
// honour it server-side. Implementations should never error on per-request
// fallible conditions (oversized query, partial truncation) — those should
// degrade gracefully so the caller can fall back to the input ordering.
// Reserve errors for hard failures (network, HTTP non-2xx, malformed
// response).
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []string, topN int) ([]RerankResult, error)
}
