package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// newTestRerankClient builds a RerankClient pointing at the test server. Counter
// is left unset by default; tests that exercise the truncation path set it
// explicitly.
func newTestRerankClient(url string) *RerankClient {
	c := openai.NewClient(
		option.WithBaseURL(url),
		option.WithAPIKey("test"),
		option.WithMaxRetries(0),
	)
	return &RerankClient{
		client: &c,
		model:  "qwen-reranker",
	}
}

// rerankResponseJSON builds a minimal response body. Pairs are (index, score),
// already sorted descending in the order given.
func rerankResponseJSON(pairs ...[2]float64) map[string]any {
	results := make([]map[string]any, len(pairs))
	for i, p := range pairs {
		results[i] = map[string]any{
			"index":           int(p[0]),
			"relevance_score": p[1],
		}
	}
	return map[string]any{
		"model":   "qwen-reranker",
		"object":  "list",
		"usage":   map[string]int{"prompt_tokens": 10, "total_tokens": 10},
		"results": results,
	}
}

func TestRerank_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path = %s, want /rerank", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		// Server returns descending order: doc 2 best, doc 0 worst — exactly the
		// shape the live llama-server emits (already sorted).
		json.NewEncoder(w).Encode(rerankResponseJSON(
			[2]float64{2, 8.6e-22},
			[2]float64{1, 3.6e-28},
			[2]float64{0, 6.8e-33},
		))
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	got, err := client.Rerank(context.Background(), "panda", []string{"hi", "bear", "panda bear"}, 3)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len(results) = %d, want 3", len(got))
	}
	if got[0].Index != 2 || got[1].Index != 1 || got[2].Index != 0 {
		t.Errorf("index order = [%d %d %d], want [2 1 0]",
			got[0].Index, got[1].Index, got[2].Index)
	}
	// Scores should pass through as-is (no normalisation). float32 conversion
	// loses precision but the magnitude is preserved.
	if got[0].Score == 0 {
		t.Errorf("top score should be non-zero, got %v", got[0].Score)
	}
}

func TestRerank_RequestBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("decoding request: %v", err)
		}
		if req["model"] != "qwen-reranker" {
			t.Errorf("model = %v, want qwen-reranker", req["model"])
		}
		if req["query"] != "panda" {
			t.Errorf("query = %v, want panda", req["query"])
		}
		docs, _ := req["documents"].([]any)
		if len(docs) != 2 {
			t.Errorf("documents len = %d, want 2", len(docs))
		}
		if topN, ok := req["top_n"].(float64); !ok || int(topN) != 5 {
			t.Errorf("top_n = %v, want 5", req["top_n"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(rerankResponseJSON([2]float64{0, 0.9}, [2]float64{1, 0.1}))
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	if _, err := client.Rerank(context.Background(), "panda", []string{"a", "b"}, 5); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
}

func TestRerank_EmptyDocs(t *testing.T) {
	// Should return early with nil result, no error, no HTTP call.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("server should not be called on empty docs")
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	got, err := client.Rerank(context.Background(), "x", nil, 5)
	if err != nil {
		t.Errorf("Rerank: %v", err)
	}
	if got != nil {
		t.Errorf("results = %v, want nil", got)
	}
}

func TestRerank_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "model unloaded", "type": "server_error"},
		})
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	_, err := client.Rerank(context.Background(), "q", []string{"a"}, 1)
	if err == nil {
		t.Fatal("expected error on 500, got nil")
	}
}

func TestRerank_DefendsAgainstOutOfRangeIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Server claims index=99 — must not crash a caller passing 2 docs.
		json.NewEncoder(w).Encode(rerankResponseJSON(
			[2]float64{99, 0.9},
			[2]float64{0, 0.5},
		))
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	got, err := client.Rerank(context.Background(), "q", []string{"a", "b"}, 2)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("results len = %d, want 1 (out-of-range index dropped)", len(got))
	}
	if got[0].Index != 0 {
		t.Errorf("got index = %d, want 0", got[0].Index)
	}
}

// stubCounter implements tokencount.Counter with deterministic per-rune
// tokenisation, giving us predictable budget arithmetic for the truncation
// tests without spinning up a real /tokenize server.
type stubCounter struct {
	tokenizeErr   error
	detokenizeErr error
}

func (s stubCounter) Count(_ context.Context, text string) (int, error) {
	return len([]rune(text)), nil
}
func (s stubCounter) Tokenize(_ context.Context, text string) ([]int, error) {
	if s.tokenizeErr != nil {
		return nil, s.tokenizeErr
	}
	runes := []rune(text)
	out := make([]int, len(runes))
	for i, r := range runes {
		out[i] = int(r)
	}
	return out, nil
}
func (s stubCounter) Detokenize(_ context.Context, tokens []int) (string, error) {
	if s.detokenizeErr != nil {
		return "", s.detokenizeErr
	}
	runes := make([]rune, len(tokens))
	for i, t := range tokens {
		runes[i] = rune(t)
	}
	return string(runes), nil
}

func TestRerank_TruncatesOversizedDocs(t *testing.T) {
	var seenDocs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Documents []string `json:"documents"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("decoding request: %v", err)
		}
		seenDocs = req.Documents

		w.Header().Set("Content-Type", "application/json")
		results := make([][2]float64, len(req.Documents))
		for i := range req.Documents {
			results[i] = [2]float64{float64(i), float64(len(req.Documents) - i)}
		}
		json.NewEncoder(w).Encode(rerankResponseJSON(results...))
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	client.counter = stubCounter{}
	client.maxTokens = 20
	client.reservedTokens = 4
	// query "panda" → 5 tokens. Budget = 20 - 5 - 4 = 11 tokens per doc.

	docs := []string{
		"short",
		"this string is way longer than eleven runes so it must be truncated",
	}
	if _, err := client.Rerank(context.Background(), "panda", docs, 2); err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	if len(seenDocs) != 2 {
		t.Fatalf("server saw %d docs, want 2", len(seenDocs))
	}
	if seenDocs[0] != "short" {
		t.Errorf("doc 0 should pass through unchanged, got %q", seenDocs[0])
	}
	if len([]rune(seenDocs[1])) != 11 {
		t.Errorf("doc 1 truncated to %d runes, want 11 (budget)", len([]rune(seenDocs[1])))
	}
}

func TestRerank_QueryOverflowReturnsPassthrough(t *testing.T) {
	// Server should not be called when the query alone overflows the budget.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("server should not be called when query overflows")
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	client.counter = stubCounter{}
	client.maxTokens = 4 // tiny window
	client.reservedTokens = 1
	// query "much longer than four tokens" — overflows; budget < 0.

	got, err := client.Rerank(context.Background(),
		"much longer than four tokens",
		[]string{"a", "b", "c"}, 3)
	if err != nil {
		t.Fatalf("Rerank: %v (should be soft fall-through, not error)", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d results, want 3 (passthrough)", len(got))
	}
	for i, r := range got {
		if r.Index != i {
			t.Errorf("passthrough[%d].Index = %d, want %d", i, r.Index, i)
		}
	}
}

func TestRerank_TokenizeErrorFailsOpen(t *testing.T) {
	// If /tokenize errors, sending the doc as-is is strictly better than
	// refusing to rerank — verify we fall through to the request unchanged.
	var seenDocs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Documents []string `json:"documents"`
		}
		_ = json.Unmarshal(body, &req)
		seenDocs = req.Documents
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(rerankResponseJSON([2]float64{0, 0.5}))
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	client.counter = stubCounter{tokenizeErr: errors.New("boom")}
	client.maxTokens = 1024

	docs := []string{"this should not be truncated"}
	if _, err := client.Rerank(context.Background(), "q", docs, 1); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(seenDocs) != 1 || seenDocs[0] != docs[0] {
		t.Errorf("server saw %v, want unchanged %v", seenDocs, docs)
	}
}

func TestRerank_TimeoutCarriedThrough(t *testing.T) {
	// A server that delays its response longer than the client's per-request
	// timeout should produce an error in well under the test deadline. The
	// handler MUST race the response against r.Context().Done() so that when
	// the client tears down the connection, the handler exits and srv.Close()
	// can complete (otherwise the test deadlocks on shutdown).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second):
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(rerankResponseJSON([2]float64{0, 0.1}))
		case <-r.Context().Done():
			return
		}
	}))
	defer srv.Close()

	client := newTestRerankClient(srv.URL)
	client.timeout = 50 * time.Millisecond

	start := time.Now()
	_, err := client.Rerank(context.Background(), "q", []string{"a"}, 1)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if elapsed > 1*time.Second {
		t.Errorf("client did not honour timeout (elapsed %v, timeout 50ms)", elapsed)
	}
}

func TestNewRerankClient_Defaults(t *testing.T) {
	c := NewRerankClient(RerankClientConfig{
		BaseURL: "http://localhost:1337/v1",
		Model:   "qwen-reranker",
	})
	if c.model != "qwen-reranker" {
		t.Errorf("model = %q, want qwen-reranker", c.model)
	}
	if c.client == nil {
		t.Error("client is nil")
	}
}
