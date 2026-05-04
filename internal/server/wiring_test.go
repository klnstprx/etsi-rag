package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"

	"github.com/rs/zerolog"
)

// testServer builds a minimal *Server for unit-testing wiring methods.
// Uses an in-memory DB and disabled logger.
func testServer(t *testing.T, cfg *config.Config) *Server {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Embedding.Dimension == 0 {
		cfg.Embedding.Dimension = 4
	}
	if err := store.Migrate(db); err != nil {
		db.Close()
		t.Fatal(err)
	}
	st := store.NewSQLiteStore(db, func() int { return cfg.Embedding.Dimension })
	t.Cleanup(func() { st.Close() })

	return &Server{
		cfg:    cfg,
		store:  st,
		logger: zerolog.New(os.Stderr).Level(zerolog.Disabled),
	}
}

func defaultConfig() *config.Config {
	return &config.Config{
		Server:   config.ServerConfig{Host: "localhost", Port: 0, ReadTimeout: 5, WriteTimeout: 30},
		Database: config.DatabaseConfig{Path: ":memory:"},
		Embedding: config.EmbeddingConfig{
			Dimension: 4,
			MaxTokens: 512,
			BatchSize: 32,
		},
		Retrieval: config.RetrievalConfig{TopK: 5, MaxContextTokens: 2048},
		Session:   config.SessionConfig{MaxMessages: 10, TTLSeconds: 3600},
		Chunking:  config.ChunkingConfig{MaxChunkSize: 256, ChunkOverlap: 32, MinChunkSize: 64},
	}
}

// --- resolveEmbedClient ---

func TestResolveEmbedClient_Found(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "local", BaseURL: "http://localhost:8080/v1", Enabled: true},
	}

	s := testServer(t, cfg)
	client, err := s.resolveEmbedClient("local", llm.EmbedClientConfig{Model: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestResolveEmbedClient_NotFound(t *testing.T) {
	s := testServer(t, defaultConfig())

	_, err := s.resolveEmbedClient("nonexistent", llm.EmbedClientConfig{})
	if err == nil {
		t.Fatal("expected error for missing backend")
	}
}

func TestResolveEmbedClient_Disabled(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "off", BaseURL: "http://localhost:8080/v1", Enabled: false},
	}

	s := testServer(t, cfg)
	_, err := s.resolveEmbedClient("off", llm.EmbedClientConfig{})
	if err == nil {
		t.Fatal("expected error for disabled backend")
	}
}

// --- resolveChatClient ---

func TestResolveChatClient_Found(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "local", BaseURL: "http://localhost:8080/v1", Enabled: true},
	}

	s := testServer(t, cfg)
	client, err := s.resolveChatClient("local", llm.ChatClientConfig{Model: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestResolveChatClient_NotFound(t *testing.T) {
	s := testServer(t, defaultConfig())

	_, err := s.resolveChatClient("nonexistent", llm.ChatClientConfig{})
	if err == nil {
		t.Fatal("expected error for missing backend")
	}
}

// --- resolveRerankClient ---

func TestResolveRerankClient_Found(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: "http://localhost:1337/v1", Enabled: true, IsLlamaCpp: true},
	}
	s := testServer(t, cfg)

	client, err := s.resolveRerankClient("llama", llm.RerankClientConfig{Model: "qwen-reranker"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestResolveRerankClient_NotFound(t *testing.T) {
	s := testServer(t, defaultConfig())
	_, err := s.resolveRerankClient("ghost", llm.RerankClientConfig{})
	if err == nil {
		t.Fatal("expected error for missing backend")
	}
}

func TestResolveRerankClient_RejectsNonLlamaCpp(t *testing.T) {
	// Defense in depth: even if config.Validate is bypassed, the resolver
	// must refuse to build a client against a non-llama-cpp backend.
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "openai", BaseURL: "https://api.openai.com/v1", Enabled: true, IsLlamaCpp: false},
	}
	s := testServer(t, cfg)

	_, err := s.resolveRerankClient("openai", llm.RerankClientConfig{})
	if err == nil {
		t.Fatal("expected error for non-llama-cpp backend")
	}
	if !strings.Contains(err.Error(), "llama") {
		t.Errorf("error should explain llama-cpp constraint, got: %v", err)
	}
}

// --- resolveCounter ---

func TestResolveCounter_Tiktoken(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "openai", BaseURL: "http://localhost/v1", Enabled: true, IsLlamaCpp: false},
	}

	s := testServer(t, cfg)
	counter, err := s.resolveCounter("openai", "gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if counter == nil {
		t.Fatal("expected non-nil counter")
	}
}

func TestResolveCounter_LlamaCpp(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: "http://localhost:8080/v1", Enabled: true, IsLlamaCpp: true},
	}

	s := testServer(t, cfg)
	counter, err := s.resolveCounter("llama", "some-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if counter == nil {
		t.Fatal("expected non-nil counter")
	}
}

func TestResolveCounter_NotFound(t *testing.T) {
	s := testServer(t, defaultConfig())

	_, err := s.resolveCounter("missing", "gpt-4o")
	if err == nil {
		t.Fatal("expected error for missing backend")
	}
}

// --- probeLlamaCppBackends ---

func TestProbeLlamaCppBackends_AppliesValues(t *testing.T) {
	// Mock llama.cpp server returning /props and /v1/models
	mux := http.NewServeMux()
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"default_generation_settings": map[string]any{"n_ctx": 4096},
			"model_path":                  "/models/test.gguf",
		})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "test-model", "meta": map[string]any{"n_embd": 768}},
			},
		})
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: ts.URL + "/v1", Enabled: true, IsLlamaCpp: true},
	}
	cfg.Embedding.Backend = "llama"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "llama"
	cfg.Agent.Model = "test-model"

	s := testServer(t, cfg)
	s.probeLlamaCppBackends()

	if cfg.Embedding.MaxTokens != 4096 {
		t.Errorf("Embedding.MaxTokens = %d, want 4096", cfg.Embedding.MaxTokens)
	}
	if cfg.Embedding.Dimension != 768 {
		t.Errorf("Embedding.Dimension = %d, want 768", cfg.Embedding.Dimension)
	}
	if cfg.Agent.ContextWindow != 4096 {
		t.Errorf("Agent.ContextWindow = %d, want 4096", cfg.Agent.ContextWindow)
	}
}

func TestProbeLlamaCppBackends_AppliesRerankerContext(t *testing.T) {
	// /props returns n_ctx=4096 — verify the probe writes that into
	// cfg.Reranker.MaxTokens so RerankClient.fitToContext has a real budget.
	mux := http.NewServeMux()
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"default_generation_settings": map[string]any{"n_ctx": 4096},
			"model_path":                  "/models/qwen-reranker.gguf",
			"model_alias":                 "qwen-reranker",
		})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"id": "qwen-reranker"}},
		})
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: ts.URL + "/v1", Enabled: true, IsLlamaCpp: true},
	}
	cfg.Reranker.Backend = "llama"
	cfg.Reranker.Model = "qwen-reranker"

	s := testServer(t, cfg)
	s.probeLlamaCppBackends()

	if cfg.Reranker.MaxTokens != 4096 {
		t.Errorf("Reranker.MaxTokens = %d, want 4096", cfg.Reranker.MaxTokens)
	}
}

func TestProbeLlamaCppBackends_NoBackends(t *testing.T) {
	cfg := defaultConfig()
	s := testServer(t, cfg)

	// Should not panic with no backends configured
	s.probeLlamaCppBackends()
}

func TestProbeLlamaCppBackends_NonLlamaCpp(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "openai", BaseURL: "http://localhost/v1", Enabled: true, IsLlamaCpp: false},
	}
	cfg.Embedding.Backend = "openai"

	s := testServer(t, cfg)
	original := cfg.Embedding.MaxTokens

	s.probeLlamaCppBackends()

	if cfg.Embedding.MaxTokens != original {
		t.Errorf("MaxTokens changed for non-llama.cpp backend: got %d, want %d", cfg.Embedding.MaxTokens, original)
	}
}

// --- dimension via config closure ---

func TestProbeSyncsStoreDimension(t *testing.T) {
	// Mock llama.cpp returning n_embd=768 via /v1/models
	mux := http.NewServeMux()
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"default_generation_settings": map[string]any{"n_ctx": 2048},
		})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "emb", "meta": map[string]any{"n_embd": 768}},
			},
		})
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: ts.URL + "/v1", Enabled: true, IsLlamaCpp: true},
	}
	cfg.Embedding.Backend = "llama"
	cfg.Embedding.Model = "emb"
	cfg.Embedding.Dimension = 4 // will be overridden by probe

	s := testServer(t, cfg)
	s.probeLlamaCppBackends()

	// Probe updates cfg.Embedding.Dimension; store reads it via closure
	ctx := context.Background()
	doc := &domain.Document{
		ID: "d1", Title: "T", Content: "C",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	s.store.CreateDocument(ctx, doc)

	if err := s.store.CreateChunks(ctx, "d1",
		[]domain.Chunk{{Content: "c", ChunkIndex: 0}},
		[][]float32{make([]float32, 768)},
	); err != nil {
		t.Errorf("768-dim embedding rejected after probe: %v", err)
	}
}

func TestConfigDimensionChange_SeenByStore(t *testing.T) {
	cfg := defaultConfig()
	cfg.Embedding.Dimension = 4
	s := testServer(t, cfg)

	ctx := context.Background()
	doc := &domain.Document{
		ID: "d1", Title: "T", Content: "C",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	s.store.CreateDocument(ctx, doc)

	// Admin UI changes dimension — store sees it immediately via closure
	cfg.Embedding.Dimension = 128

	err := s.store.CreateChunks(ctx, "d1",
		[]domain.Chunk{{Content: "c", ChunkIndex: 0}},
		[][]float32{{1, 0, 0, 0}},
	)
	if err == nil {
		t.Error("expected 4-dim embedding to be rejected after dimension changed to 128")
	}

	if err := s.store.CreateChunks(ctx, "d1",
		[]domain.Chunk{{Content: "c", ChunkIndex: 0}},
		[][]float32{make([]float32, 128)},
	); err != nil {
		t.Errorf("128-dim embedding rejected after config change: %v", err)
	}
}

// --- attachReranker ---

// testServerWithPipeline builds a Server with a real Pipeline so attachReranker
// has somewhere to write SetReranker calls.
func testServerWithPipeline(t *testing.T, cfg *config.Config) *Server {
	t.Helper()
	s := testServer(t, cfg)
	chunker := chunking.NewRecursiveCharacterChunker(cfg.Chunking)
	s.pipeline = rag.NewPipeline(s.store, nil, chunker, nil, cfg.Retrieval)
	return s
}

func TestAttachReranker_DisabledIsNoop(t *testing.T) {
	cfg := defaultConfig()
	cfg.Reranker.Enabled = false
	s := testServerWithPipeline(t, cfg)

	s.attachReranker() // must not panic, must not error
}

func TestAttachReranker_BadBackendDegradesGracefully(t *testing.T) {
	cfg := defaultConfig()
	cfg.Reranker.Enabled = true
	cfg.Reranker.Backend = "ghost"
	cfg.Reranker.Model = "qwen-reranker"
	s := testServerWithPipeline(t, cfg)

	// Must not panic — the safeguards in attachReranker should swallow the
	// resolution failure and disable the reranker on the pipeline.
	s.attachReranker()
}

func TestAttachReranker_LlamaCppBackendInstalls(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: "http://localhost:1337/v1", Enabled: true, IsLlamaCpp: true},
	}
	cfg.Reranker.Enabled = true
	cfg.Reranker.Backend = "llama"
	cfg.Reranker.Model = "qwen-reranker"
	cfg.Reranker.TopN = 30
	cfg.Reranker.MaxTokens = 4096
	cfg.Reranker.ReservedTokens = 16
	s := testServerWithPipeline(t, cfg)

	s.attachReranker() // success path — installs on the pipeline
}

// --- agentSystemPrompt ---

func TestAgentSystemPrompt_Default(t *testing.T) {
	s := testServer(t, defaultConfig())

	result := s.agentSystemPrompt()
	if result == "" {
		t.Fatal("expected non-empty prompt")
	}
	// Default template has "0 documents" when DB is empty
	if !strings.Contains(result, "0 documents") {
		t.Errorf("expected '0 documents' in default prompt, got: %s", result)
	}
}

func TestAgentSystemPrompt_CustomTemplate(t *testing.T) {
	cfg := defaultConfig()
	cfg.Agent.SystemPrompt = "KB has {{.DocumentCount}} docs."

	s := testServer(t, cfg)
	result := s.agentSystemPrompt()
	if result != "KB has 0 docs." {
		t.Errorf("got %q, want %q", result, "KB has 0 docs.")
	}
}

func TestAgentSystemPrompt_WithDocuments(t *testing.T) {
	cfg := defaultConfig()
	s := testServer(t, cfg)

	// Insert a document + chunk to get nonzero counts
	doc := &domain.Document{ID: "doc1", Title: "Test", Content: "hello", Language: domain.LanguageES}
	if err := s.store.CreateDocument(context.Background(), doc); err != nil {
		t.Fatal(err)
	}
	emb := [][]float32{{0.1, 0.2, 0.3, 0.4}}
	chunks := []domain.Chunk{{DocID: "doc1", Content: "hello", ChunkIndex: 0}}
	if err := s.store.CreateChunks(context.Background(), "doc1", chunks, emb); err != nil {
		t.Fatal(err)
	}

	result := s.agentSystemPrompt()
	if !strings.Contains(result, "1 documents") {
		t.Errorf("expected '1 documents' in prompt, got: %s", result)
	}
	if !strings.Contains(result, "1 searchable chunks") {
		t.Errorf("expected '1 searchable chunks' in prompt, got: %s", result)
	}
}

func TestAgentSystemPrompt_InvalidTemplate(t *testing.T) {
	cfg := defaultConfig()
	cfg.Agent.SystemPrompt = "broken {{.Unclosed"

	s := testServer(t, cfg)
	result := s.agentSystemPrompt()

	// Should fall back to raw template text
	if result != cfg.Agent.SystemPrompt {
		t.Errorf("expected raw template fallback, got: %s", result)
	}
}

func TestAgentSystemPrompt_InvalidTemplateFallsBackToDefault(t *testing.T) {
	cfg := defaultConfig()
	cfg.Agent.SystemPrompt = ""

	s := testServer(t, cfg)
	result := s.agentSystemPrompt()

	// Should contain text from the default template
	if !strings.Contains(result, "ETSI") {
		t.Errorf("expected default prompt content, got: %s", result)
	}
}

func TestAgentSystemPrompt_PlainTextPassthrough(t *testing.T) {
	cfg := defaultConfig()
	cfg.Agent.SystemPrompt = "You are a helpful bot."

	s := testServer(t, cfg)
	result := s.agentSystemPrompt()
	if result != "You are a helpful bot." {
		t.Errorf("got %q, want %q", result, "You are a helpful bot.")
	}
}

func TestAgentSystemPrompt_DefaultTemplateString(t *testing.T) {
	// Verify prompts.Agent is accessible and non-empty
	if prompts.Agent == "" {
		t.Fatal("prompts.Agent should not be empty")
	}
}
