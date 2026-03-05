package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
)

// mockCounter counts words as tokens for testing.
type mockCounter struct{}

func (m *mockCounter) Count(_ context.Context, text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	return len(strings.Fields(text)), nil
}

func (m *mockCounter) Tokenize(_ context.Context, text string) ([]int, error) {
	words := strings.Fields(text)
	tokens := make([]int, len(words))
	for i := range tokens {
		tokens[i] = i
	}
	return tokens, nil
}

func (m *mockCounter) Detokenize(_ context.Context, tokens []int) (string, error) {
	parts := make([]string, len(tokens))
	for i := range parts {
		parts[i] = "word"
	}
	return strings.Join(parts, " "), nil
}

// mockLLM implements llm.LLM for handler tests.
type mockLLM struct{}

func (m *mockLLM) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.5, 0.5, 0.0, 0.0}, nil
}

func (m *mockLLM) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = []float32{0.5, 0.5, 0.0, 0.0}
	}
	return result, nil
}

func (m *mockLLM) Chat(_ context.Context, _ []openai.ChatCompletionMessageParamUnion) (string, error) {
	return "Mock response", nil
}

func (m *mockLLM) ChatStream(_ context.Context, _ []openai.ChatCompletionMessageParamUnion) (<-chan string, <-chan error) {
	tokens := make(chan string)
	errc := make(chan error)
	close(tokens)
	close(errc)
	return tokens, errc
}

func setupTestHandler(t *testing.T) (*Handler, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Migrate(db, 4); err != nil {
		db.Close()
		t.Fatal(err)
	}

	s := store.NewSQLiteStore(db, 4)
	mock := &mockLLM{}
	counter := &mockCounter{}
	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{MaxChunkSize: 100})
	pipeline := rag.NewPipeline(s, mock, mock, chunker, counter, config.RAGConfig{
		TopK:                5,
		SimilarityThreshold: 0.0,
		MaxContextTokens:    1024,
		SystemPromptES:      "Test system prompt",
		SystemPromptEN:      "Test system prompt EN",
	})
	sessionStore := session.NewStore(config.SessionConfig{MaxMessages: 10, TTLSeconds: 3600})
	t.Cleanup(sessionStore.Close)

	h := NewHandler(HandlerConfig{
		EmbeddingClient: mock,
		ChatClient:      mock,
		Counter:         counter,
		Pipeline:        pipeline,
		Store:           s,
		SessionStore:    sessionStore,
	})

	return h, s
}

func TestHealthCheck(t *testing.T) {
	h, _ := setupTestHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	h.HealthCheck(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got %q", resp.Status)
	}
}

func TestIngestAndListDocuments(t *testing.T) {
	h, _ := setupTestHandler(t)

	// Ingest
	body, _ := json.Marshal(IngestRequest{
		Documents: []DocumentInput{
			{
				ID:       "doc-1",
				Title:    "Test Doc",
				Content:  "Test content for ingestion.",
				Language: "es",
				Category: "general",
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Ingest(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}

	var ingestResp IngestResponse
	json.NewDecoder(w.Body).Decode(&ingestResp)
	if ingestResp.IngestedCount != 1 {
		t.Errorf("expected 1 ingested, got %d", ingestResp.IngestedCount)
	}

	// List
	req = httptest.NewRequest(http.MethodGet, "/v1/rag/documents", nil)
	w = httptest.NewRecorder()

	h.ListDocuments(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var docs []domain.Document
	json.NewDecoder(w.Body).Decode(&docs)
	if len(docs) != 1 {
		t.Errorf("expected 1 document, got %d", len(docs))
	}
}

func TestDeleteDocument(t *testing.T) {
	h, s := setupTestHandler(t)
	ctx := context.Background()

	// Create a document first
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test",
		Content:  "Content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}

	// Delete via handler (need chi context for URL params)
	req := httptest.NewRequest(http.MethodDelete, "/v1/rag/documents/doc-1", nil)
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add("id", "doc-1")
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))

	w := httptest.NewRecorder()
	h.DeleteDocument(w, req)

	if w.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d: %s", w.Code, w.Body.String())
	}
}

func TestDeleteDocumentNotFound(t *testing.T) {
	h, _ := setupTestHandler(t)

	req := httptest.NewRequest(http.MethodDelete, "/v1/rag/documents/nonexistent", nil)
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add("id", "nonexistent")
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))

	w := httptest.NewRecorder()
	h.DeleteDocument(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestIngestInvalidLanguage(t *testing.T) {
	h, _ := setupTestHandler(t)

	body, _ := json.Marshal(IngestRequest{
		Documents: []DocumentInput{
			{
				ID:       "doc-1",
				Title:    "Test",
				Content:  "Content",
				Language: "fr",
				Category: "general",
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Ingest(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestRAGChat(t *testing.T) {
	h, s := setupTestHandler(t)
	ctx := context.Background()

	// Insert test data
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Doc",
		Content:  "Relevant info",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	s.CreateDocument(ctx, doc)
	s.CreateChunks(ctx, "doc-1",
		[]domain.Chunk{{Content: "Relevant info", ChunkIndex: 0}},
		[][]float32{{0.5, 0.5, 0.0, 0.0}},
	)

	body, _ := json.Marshal(RAGChatRequest{
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Tell me about this"),
					},
				},
			},
		},
		Model:          "test-model",
		IncludeSources: true,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/rag/chat", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.RAGChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp RAGChatResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Choices) == 0 {
		t.Fatal("expected choices in response")
	}
	if resp.Choices[0].Message.Content == "" {
		t.Error("expected non-empty content")
	}
}
