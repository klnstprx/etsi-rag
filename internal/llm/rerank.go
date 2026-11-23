package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/openai/openai-go"

	"etsi-rag/internal/tokencount"
)

// Compile-time check.
var _ Reranker = (*RerankClient)(nil)

// RerankClientConfig holds everything needed to construct a reranker client.
//
// Counter is optional: when non-nil and MaxTokens > 0, RerankClient truncates
// each document so the cross-encoder's context window can hold (query + doc +
// reserved). When nil, documents are sent as-is.
type RerankClientConfig struct {
	BaseURL        string
	APIKey         string
	Model          string
	Counter        tokencount.Counter
	MaxTokens      int
	ReservedTokens int
	Timeout        time.Duration
}

// RerankClient wraps the OpenAI client to call llama.cpp's /rerank endpoint.
// /rerank is a llama.cpp / Jina extension and not part of the OpenAI surface,
// so we use the SDK's documented escape hatch (Client.Post) instead of
// hand-rolling net/http.
type RerankClient struct {
	client         *openai.Client
	model          string
	counter        tokencount.Counter
	maxTokens      int
	reservedTokens int
	timeout        time.Duration
}

// NewRerankClient creates a new reranker client.
func NewRerankClient(cfg RerankClientConfig) *RerankClient {
	return &RerankClient{
		client:         newOpenAI(cfg.BaseURL, cfg.APIKey),
		model:          cfg.Model,
		counter:        cfg.Counter,
		maxTokens:      cfg.MaxTokens,
		reservedTokens: cfg.ReservedTokens,
		timeout:        cfg.Timeout,
	}
}

// rerankRequest is the body sent to /rerank. The shape matches the
// llama.cpp / Jina convention; Cohere clients accept the same.
type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// rerankResponse decodes the part of llama-server's /rerank reply we use
// (verified 2026-05-04 against the router build with qwen-reranker):
//
//	{"model":"qwen-reranker","object":"list",
//	 "usage":{...},
//	 "results":[{"index":2,"relevance_score":8.6e-22}, ...]}
type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank scores each document against the query and returns them in
// descending relevance order. The returned RerankResult.Index refers back to
// the position in the input docs slice so the caller can reorder its own
// candidate list without round-tripping document identifiers.
//
// Soft failures (query alone exceeds the model's context, empty docs)
// return the input ordering with a nil error so the caller can keep going.
// Hard failures (network, HTTP non-2xx, malformed response) surface as
// errors.
func (c *RerankClient) Rerank(ctx context.Context, query string, docs []string, topN int) ([]RerankResult, error) {
	if len(docs) == 0 {
		return nil, nil
	}

	if c.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	fitted, ok := c.fitToContext(ctx, query, docs)
	if !ok {
		// Query alone overflows the cross-encoder's context window — the
		// reranker can't help. Hand back the original ordering.
		return passthrough(len(docs)), nil
	}

	body := rerankRequest{
		Model:     c.model,
		Query:     query,
		Documents: fitted,
		TopN:      topN,
	}
	var resp rerankResponse
	if err := c.client.Post(ctx, "rerank", body, &resp); err != nil {
		return nil, fmt.Errorf("rerank request: %w", err)
	}

	out := make([]RerankResult, 0, len(resp.Results))
	for _, r := range resp.Results {
		if r.Index < 0 || r.Index >= len(docs) {
			continue // defensive: malformed response shouldn't crash callers
		}
		out = append(out, RerankResult{Index: r.Index, Score: float32(r.RelevanceScore)})
	}
	return out, nil
}

// fitToContext truncates each document so (query + doc + reserved) fits in
// the cross-encoder's context window. Returns (truncatedDocs, true) on
// success; (nil, false) when even the query alone overflows the budget — the
// caller should pass through the original ordering in that case.
//
// When Counter or MaxTokens is unset the function is a no-op (docs are
// returned unchanged), matching the behaviour callers get when the reranker
// backend exposes no /tokenize endpoint.
func (c *RerankClient) fitToContext(ctx context.Context, query string, docs []string) ([]string, bool) {
	if c.counter == nil || c.maxTokens <= 0 {
		return docs, true
	}

	qTok, err := c.counter.Tokenize(ctx, query)
	if err != nil {
		// Fail open — sending the doc as-is is no worse than refusing to rerank.
		return docs, true
	}
	budget := c.maxTokens - len(qTok) - c.reservedTokens
	if budget <= 0 {
		return nil, false
	}

	out := make([]string, len(docs))
	for i, d := range docs {
		toks, err := c.counter.Tokenize(ctx, d)
		if err != nil || len(toks) <= budget {
			out[i] = d
			continue
		}
		truncated, err := c.counter.Detokenize(ctx, toks[:budget])
		if err != nil {
			out[i] = d
			continue
		}
		out[i] = truncated
	}
	return out, true
}

// passthrough returns RerankResults preserving the input ordering. Used as the
// soft-fallback when the reranker can't help (query overflow, empty docs).
func passthrough(n int) []RerankResult {
	out := make([]RerankResult, n)
	for i := range out {
		out[i] = RerankResult{Index: i, Score: 0}
	}
	return out
}
