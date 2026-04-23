package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// mockEmbedder implements llm.Embedder for handler tests.
type mockEmbedder struct{}

func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.5, 0.5, 0.0, 0.0}, nil
}

func (m *mockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = []float32{0.5, 0.5, 0.0, 0.0}
	}
	return result, nil
}

func setupTestHandler(t *testing.T) (*Handler, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	if err := store.Migrate(db); err != nil {
		t.Fatal(err)
	}

	s := store.NewSQLiteStore(db, func() int { return 4 })
	mock := &mockEmbedder{}
	counter := &testutil.MockCounter{}
	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{MaxChunkSize: 100})
	pipeline := rag.NewPipeline(s, mock, chunker, counter, config.RAGConfig{
		TopK:             5,
		MaxContextTokens: 1024,
	})
	sessionStore := session.NewStore(config.SessionConfig{MaxMessages: 10, TTLSeconds: 3600})
	t.Cleanup(sessionStore.Close)

	h := NewHandler(HandlerConfig{
		EmbeddingClient: mock,
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
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
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
	if err := json.NewDecoder(w.Body).Decode(&ingestResp); err != nil {
		t.Fatalf("decoding ingest response: %v", err)
	}
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
	if err := json.NewDecoder(w.Body).Decode(&docs); err != nil {
		t.Fatalf("decoding documents: %v", err)
	}
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

func TestChatCompletions_NoAgent(t *testing.T) {
	h, _ := setupTestHandler(t)

	body, _ := json.Marshal(openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("hello"),
					},
				},
			},
		},
		Model: "test",
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.ChatCompletions(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when agent is nil, got %d", w.Code)
	}
}

func TestChatCompletions_SessionMode_RejectsMultipleMessages(t *testing.T) {
	// This test validates the validation logic in extractSingleUserMessage directly.

	tests := []struct {
		name     string
		messages []openai.ChatCompletionMessageParamUnion
		wantErr  bool
	}{
		{
			name:     "empty messages",
			messages: nil,
			wantErr:  true,
		},
		{
			name: "system message rejected",
			messages: []openai.ChatCompletionMessageParamUnion{
				{OfSystem: &openai.ChatCompletionSystemMessageParam{
					Content: openai.ChatCompletionSystemMessageParamContentUnion{
						OfString: openai.String("system"),
					},
				}},
			},
			wantErr: true,
		},
		{
			name: "single user message accepted",
			messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("hello"),
					},
				}},
			},
			wantErr: false,
		},
		{
			name: "two user messages rejected",
			messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("hello"),
					},
				}},
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("world"),
					},
				}},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := extractSingleUserMessage(tt.messages)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractSingleUserMessage() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// fakeChatCompletionServer returns an httptest.Server that responds with
// a canned chat completion (no tool calls).
func fakeChatCompletionServer(t *testing.T, responseContent string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"model":   "test-model",
			"choices": []map[string]any{
				{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]any{
						"role":    "assistant",
						"content": responseContent,
					},
				},
			},
			"usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestChatCompletions_SessionMode_HappyPath(t *testing.T) {
	h, _ := setupTestHandler(t)

	// Stand up a fake LLM server
	srv := fakeChatCompletionServer(t, "Hello from agent")
	defer srv.Close()

	// Build a real agent sharing the handler's session store
	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL:     srv.URL,
		Model:       "test-model",
		MaxTokens:   100,
		Temperature: 0,
	})

	ag := agent.NewAgent(chatClient, nil, "You are a test assistant.", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.SetAgent(ag)

	// Create a session in the shared store
	sessionID := h.sessionStore.Create()

	// Send a chat completion request
	body, _ := json.Marshal(ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hi there"),
				},
			}},
		},
		Model: "test-model",
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("X-Session-ID", sessionID)
	w := httptest.NewRecorder()

	h.ChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response content
	var resp ChatCompletionWithCitations
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
	if resp.Choices[0].Message.Content != "Hello from agent" {
		t.Errorf("got content %q, want %q", resp.Choices[0].Message.Content, "Hello from agent")
	}

	// Verify session ID header
	if got := w.Header().Get("X-Session-ID"); got != sessionID {
		t.Errorf("X-Session-ID header = %q, want %q", got, sessionID)
	}

	// Verify messages were persisted to session (user + assistant = 2)
	sess := h.sessionStore.Get(sessionID)
	if sess == nil {
		t.Fatal("session should exist after chat")
	}
	msgs := sess.GetMessages()
	if len(msgs) != 2 {
		t.Errorf("session has %d messages, want 2 (user + assistant)", len(msgs))
	}
}
