package tokencount

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestLlamaCppCounterRetriesOn5xx(t *testing.T) {
	// Flaky upstream: first 2 calls return 500, 3rd succeeds. Mirrors a
	// llama-server crash window
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(tokenizeResponse{Tokens: []int{1, 2, 3}})
	}))
	defer srv.Close()

	c := NewLlamaCppCounter(srv.URL, "test-model", nil)
	tokens, err := c.Tokenize(context.Background(), "anything")
	if err != nil {
		t.Fatalf("expected retry to succeed, got: %v", err)
	}
	if len(tokens) != 3 {
		t.Errorf("expected 3 tokens after retry, got %d", len(tokens))
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("expected 3 attempts, got %d", got)
	}
}

func TestLlamaCppCounter(t *testing.T) {
	// Mock llama.cpp server
	mux := http.NewServeMux()
	mux.HandleFunc("POST /tokenize", func(w http.ResponseWriter, r *http.Request) {
		var req tokenizeRequest
		json.NewDecoder(r.Body).Decode(&req)
		// Simple mock: return tokens based on content length/4
		mockTokens := make([]int, len(req.Content)/4)
		for i := range mockTokens {
			mockTokens[i] = i + 100
		}
		if len(req.Content) > 0 && len(mockTokens) == 0 {
			mockTokens = []int{100}
		}
		json.NewEncoder(w).Encode(tokenizeResponse{Tokens: mockTokens})
	})
	mux.HandleFunc("POST /detokenize", func(w http.ResponseWriter, r *http.Request) {
		var req detokenizeRequest
		json.NewDecoder(r.Body).Decode(&req)
		json.NewEncoder(w).Encode(detokenizeResponse{Content: "detokenized"})
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	ctx := context.Background()
	c := NewLlamaCppCounter(srv.URL, "test-model", nil)

	// Test Count
	count, err := c.Count(ctx, "hello world test")
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count == 0 {
		t.Error("expected non-zero count")
	}

	// Test Tokenize
	tokens, err := c.Tokenize(ctx, "hello world test")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(tokens) == 0 {
		t.Error("expected non-empty tokens")
	}

	// Test Detokenize
	text, err := c.Detokenize(ctx, []int{100, 101, 102})
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	if text != "detokenized" {
		t.Errorf("expected 'detokenized', got %q", text)
	}
}

func TestLlamaCppCounterDetokenizeEmpty(t *testing.T) {
	// Regression: llama-server's /detokenize returns 400 for empty tokens arrays.
	// Short-circuit empty slices and do not make a round-trip at all.
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewLlamaCppCounter(srv.URL, "test-model", nil)
	text, err := c.Detokenize(context.Background(), nil)
	if err != nil {
		t.Fatalf("empty Detokenize should not error: %v", err)
	}
	if text != "" {
		t.Errorf("expected empty text, got %q", text)
	}
	if called {
		t.Error("HTTP call should have been short-circuited for empty tokens")
	}
}

// TestLlamaCppCounterError checks which failures count as the backend being
// down (ErrBackend) and which reject the text itself, since the crawler's
// embed-failure breaker counts only the former.
func TestLlamaCppCounterError(t *testing.T) {
	status := func(code int) string {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(code)
		}))
		t.Cleanup(srv.Close)
		return srv.URL
	}
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()

	tests := []struct {
		name        string
		url         string
		wantBackend bool
	}{
		{"5xx after retries", status(http.StatusInternalServerError), true},
		{"connection refused", closed.URL, true},
		{"4xx rejection", status(http.StatusBadRequest), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Disable retries so the test fails fast instead of waiting through backoff.
			c := NewLlamaCppCounter(tt.url, "test-model", nil).WithMaxRetries(0)

			_, err := c.Count(context.Background(), "test")
			if err == nil {
				t.Fatal("expected an error")
			}
			if got := errors.Is(err, ErrBackend); got != tt.wantBackend {
				t.Errorf("errors.Is(err, ErrBackend) = %v, want %v (err: %v)", got, tt.wantBackend, err)
			}
		})
	}
}

func TestTiktokenCounter(t *testing.T) {
	ctx := context.Background()

	// gpt-4 is a known model in tiktoken
	c, err := NewTiktokenCounter("gpt-4")
	if err != nil {
		t.Fatalf("NewTiktokenCounter: %v", err)
	}

	// Test Count
	count, err := c.Count(ctx, "Hello, world!")
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count == 0 {
		t.Error("expected non-zero token count")
	}

	// Test roundtrip
	tokens, err := c.Tokenize(ctx, "Hello, world!")
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != count {
		t.Errorf("Tokenize len=%d != Count=%d", len(tokens), count)
	}

	text, err := c.Detokenize(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hello, world!" {
		t.Errorf("roundtrip failed: got %q", text)
	}
}

func TestTiktokenCounterBadModel(t *testing.T) {
	_, err := NewTiktokenCounter("nonexistent-model-xyz")
	if err == nil {
		t.Error("expected error for unknown model")
	}
}
