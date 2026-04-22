package tokencount

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

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

func TestLlamaCppCounterError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	ctx := context.Background()
	c := NewLlamaCppCounter(srv.URL, "test-model", nil)

	_, err := c.Count(ctx, "test")
	if err == nil {
		t.Error("expected error for 500 response")
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
