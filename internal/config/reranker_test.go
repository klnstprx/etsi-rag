package config

import (
	"strings"
	"testing"
	"time"
)

// rerankerBackends returns a TOML stanza with one llama-cpp backend and one
// non-llama-cpp backend so reranker.backend resolution can be exercised both
// ways.
const rerankerBackends = `
[[backends]]
name = "local-llama"
base_url = "http://localhost:1337/v1"
enabled = true
is_llama_cpp = true

[[backends]]
name = "remote-openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-x"
enabled = true
is_llama_cpp = false
`

func TestRerankerDisabledByDefault(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Reranker.Enabled {
		t.Errorf("reranker.enabled = true, want false (default)")
	}
}

func TestRerankerAppliesDefaults(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Reranker.TopN != 30 {
		t.Errorf("reranker.top_n = %d, want 30", cfg.Reranker.TopN)
	}
	if cfg.Reranker.ReservedTokens != 16 {
		t.Errorf("reranker.reserved_tokens = %d, want 16", cfg.Reranker.ReservedTokens)
	}
	if cfg.Reranker.Timeout != 30*time.Second {
		t.Errorf("reranker.timeout = %s, want 30s", cfg.Reranker.Timeout)
	}
}

func TestRerankerDisabledSkipsValidation(t *testing.T) {
	// enabled=false: empty backend / model / undersized top_n should NOT block load.
	cfg, err := Load(writeTempConfig(t, `
[reranker]
enabled = false
backend = ""
model = ""
top_n = 1
`))
	if err != nil {
		t.Fatalf("Load with disabled reranker: %v", err)
	}
	if cfg.Reranker.Enabled {
		t.Error("reranker.enabled should be false")
	}
}

func TestRerankerEnabledRequiresBackend(t *testing.T) {
	_, err := Load(writeTempConfig(t, `
[reranker]
enabled = true
backend = ""
model = "qwen-reranker"
`))
	if err == nil {
		t.Fatal("expected error when reranker.enabled with empty backend")
	}
	if !strings.Contains(err.Error(), "reranker.backend") {
		t.Errorf("error should mention reranker.backend, got: %v", err)
	}
}

func TestRerankerEnabledRequiresModel(t *testing.T) {
	_, err := Load(writeTempConfig(t, rerankerBackends+`
[reranker]
enabled = true
backend = "local-llama"
model = ""
`))
	if err == nil {
		t.Fatal("expected error when reranker.enabled with empty model")
	}
	if !strings.Contains(err.Error(), "reranker.model") {
		t.Errorf("error should mention reranker.model, got: %v", err)
	}
}

func TestRerankerEnabledRequiresKnownBackend(t *testing.T) {
	_, err := Load(writeTempConfig(t, rerankerBackends+`
[reranker]
enabled = true
backend = "ghost"
model = "qwen-reranker"
`))
	if err == nil {
		t.Fatal("expected error when reranker.backend names an unknown backend")
	}
	if !strings.Contains(err.Error(), "ghost") {
		t.Errorf("error should name the offending backend, got: %v", err)
	}
}

func TestRerankerEnabledRequiresLlamaCppBackend(t *testing.T) {
	_, err := Load(writeTempConfig(t, rerankerBackends+`
[reranker]
enabled = true
backend = "remote-openai"
model = "qwen-reranker"
`))
	if err == nil {
		t.Fatal("expected error when reranker backend is not llama-cpp")
	}
	if !strings.Contains(err.Error(), "llama") {
		t.Errorf("error should explain llama.cpp constraint, got: %v", err)
	}
	if !strings.Contains(err.Error(), "remote-openai") {
		t.Errorf("error should name the offending backend, got: %v", err)
	}
}

func TestRerankerEnabledRequiresTopNGteTopK(t *testing.T) {
	_, err := Load(writeTempConfig(t, rerankerBackends+`
[retrieval]
top_k = 10

[reranker]
enabled = true
backend = "local-llama"
model = "qwen-reranker"
top_n = 5
`))
	if err == nil {
		t.Fatal("expected error when reranker.top_n < retrieval.top_k")
	}
	if !strings.Contains(err.Error(), "top_n") || !strings.Contains(err.Error(), "top_k") {
		t.Errorf("error should mention both top_n and top_k, got: %v", err)
	}
}

func TestRerankerEnabledHappyPath(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, rerankerBackends+`
[retrieval]
top_k = 10

[reranker]
enabled = true
backend = "local-llama"
model = "qwen-reranker"
top_n = 30
max_tokens = 4096
reserved_tokens = 16
timeout = "30s"
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.Reranker.Enabled {
		t.Error("reranker.enabled should be true")
	}
	if cfg.Reranker.Backend != "local-llama" {
		t.Errorf("reranker.backend = %q, want local-llama", cfg.Reranker.Backend)
	}
	if cfg.Reranker.Model != "qwen-reranker" {
		t.Errorf("reranker.model = %q, want qwen-reranker", cfg.Reranker.Model)
	}
	if cfg.Reranker.TopN != 30 {
		t.Errorf("reranker.top_n = %d, want 30", cfg.Reranker.TopN)
	}
	if cfg.Reranker.MaxTokens != 4096 {
		t.Errorf("reranker.max_tokens = %d, want 4096", cfg.Reranker.MaxTokens)
	}
}
