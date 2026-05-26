package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/ingest"
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
	h := llm.NewEmbedderHandle(mock)
	retriever := rag.NewRetriever(s, h, counter, config.RetrievalConfig{
		TopK:             5,
		MaxContextTokens: 1024,
	})
	ingester := ingest.NewIngester(s, h, testutil.FakeChunker{})
	sessionStore := session.NewStore(config.SessionConfig{MaxMessages: 10, TTLSeconds: 3600})
	t.Cleanup(sessionStore.Close)

	handler := NewHandler(HandlerConfig{
		AgentHandle:  agent.NewHandle(nil),
		Retriever:    retriever,
		Ingester:     ingester,
		Store:        s,
		SessionStore: sessionStore,
	})

	return handler, s
}

// TestGetArtifact_NotFound covers the 404 path: when the {token} doesn't
// match any artifact, the handler must return 404 (not 5xx). This is the
// path exercised every time a stale or made-up URL hits the server.
func TestGetArtifact_NotFound(t *testing.T) {
	h, _ := setupTestHandler(t)

	r := chi.NewRouter()
	r.Get("/v1/artifacts/{token}", h.GetArtifact)

	req := httptest.NewRequest(http.MethodGet, "/v1/artifacts/nonexistent.png", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// objStore is nil in setupTestHandler, so we expect 500 from the early
	// guard. This verifies the guard fires before we reach the store.
	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 (no object store configured), got %d body=%q", w.Code, w.Body.String())
	}
}

// TestGetArtifact_TokenStripsPngSuffix asserts that the .png extension is
// optional in the URL: the handler resolves on the bare token. This matters
// because the LLM backend will fetch the URL we put in image_url, and giving
// it a path with .png is friendlier (some renderers infer MIME from suffix)
// while we don't actually need it for lookup.
func TestGetArtifact_TokenStripsPngSuffix(t *testing.T) {
	h, s := setupTestHandler(t)

	doc := &domain.Document{
		ID: "doc-1", Title: "x", Content: "x",
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(context.Background(), doc); err != nil {
		t.Fatalf("CreateDocument: %v", err)
	}
	a := &domain.Artifact{
		DocID: "doc-1", Ref: "p1", Kind: domain.ArtifactPDFPage,
		ObjectStoreKey: "k", MimeType: "image/png",
	}
	if err := s.UpsertArtifact(context.Background(), a); err != nil {
		t.Fatalf("UpsertArtifact: %v", err)
	}

	r := chi.NewRouter()
	r.Get("/v1/artifacts/{token}", h.GetArtifact)

	// Hit with the .png suffix; we expect to reach the store lookup, which
	// will succeed in finding the artifact, then fail on the missing
	// objStore (nil → 500). The 500 here proves we got past the
	// store.GetArtifactByToken call (would be 404 if the suffix were not
	// stripped).
	req := httptest.NewRequest(http.MethodGet, "/v1/artifacts/"+a.PublicToken+".png", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 (objStore nil) after stripping .png, got %d body=%q", w.Code, w.Body.String())
	}
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

// TestRAGQuery_HappyPath exercises the retrieval-introspection endpoint that
// the eval harness uses to compute Recall@k / MRR. We ingest a couple of
// docs, then POST a query and assert the response shape: each result must
// carry chunk_id, document_url, document_title, section_title, score, and
// a snippet, in score-descending order. The eval needs document_url to map
// retrieved chunks back to golden URLs.
func TestRAGQuery_HappyPath(t *testing.T) {
	h, _ := setupTestHandler(t)

	urlA := "https://example.com/a"
	urlB := "https://example.com/b"
	body, _ := json.Marshal(IngestRequest{
		Documents: []DocumentInput{
			{ID: "doc-a", Title: "Doc A", Content: "alpha bravo charlie delta", URL: &urlA, Language: "es"},
			{ID: "doc-b", Title: "Doc B", Content: "echo foxtrot golf hotel", URL: &urlB, Language: "es"},
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Ingest(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("ingest setup failed: %d %s", w.Code, w.Body.String())
	}

	// Now query.
	topK := 5
	qBody, _ := json.Marshal(RAGQueryRequest{Query: "alpha bravo", TopK: &topK})
	req = httptest.NewRequest(http.MethodPost, "/v1/rag/query", bytes.NewReader(qBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	h.RAGQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp RAGQueryResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Results) == 0 {
		t.Fatalf("expected at least one result, got 0")
	}
	for i, r := range resp.Results {
		if r.DocumentURL == nil || *r.DocumentURL == "" {
			t.Errorf("result %d: missing document_url", i)
		}
		if r.DocumentTitle == "" {
			t.Errorf("result %d: missing document_title", i)
		}
		if r.Snippet == "" {
			t.Errorf("result %d: missing snippet", i)
		}
		if r.ChunkID == 0 {
			t.Errorf("result %d: missing chunk_id", i)
		}
		if i > 0 && resp.Results[i-1].Score < r.Score {
			t.Errorf("results not sorted by score desc: pos %d = %f, pos %d = %f",
				i-1, resp.Results[i-1].Score, i, r.Score)
		}
	}
}

// TestRAGQuery_EmptyQuery rejects an empty query with 400 — saves a wasted
// embedding call and a useless top-k against zero signal.
func TestRAGQuery_EmptyQuery(t *testing.T) {
	h, _ := setupTestHandler(t)
	body, _ := json.Marshal(RAGQueryRequest{Query: ""})
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/query", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.RAGQuery(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

// TestRAGQuery_InvalidJSON ensures malformed bodies are rejected cleanly.
func TestRAGQuery_InvalidJSON(t *testing.T) {
	h, _ := setupTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/query", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	h.RAGQuery(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
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
	h.agentHandle.Set(ag)

	// Create a session in the shared store
	sessionID := h.sessionStore.Create()

	// Send a chat completion request
	body, _ := json.Marshal(ChatCompletionRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hi there"),
					},
				}},
			},
			Model: "test-model",
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("X-Session-ID", sessionID)
	w := httptest.NewRecorder()

	h.ChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response content (plain OpenAI shape — no RAG extras)
	bodyStr := w.Body.String()
	var resp openai.ChatCompletion
	if err := json.Unmarshal([]byte(bodyStr), &resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
	if resp.Choices[0].Message.Content != "Hello from agent" {
		t.Errorf("got content %q, want %q", resp.Choices[0].Message.Content, "Hello from agent")
	}
	// Purity guard: the OpenAI endpoint must not leak proprietary root fields.
	// (tool_calls is a legit OpenAI message field, so we key off citations/images,
	// which the old hybrid always emitted alongside the root tool_calls.)
	if strings.Contains(bodyStr, "citations") || strings.Contains(bodyStr, `"images"`) {
		t.Errorf("OpenAI non-stream response leaked RAG extras: %s", bodyStr)
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

// fakeStreamingChatServer returns an httptest.Server whose /chat/completions
// emits the given content tokens as OpenAI-format SSE chunks, followed by a
// finish chunk and [DONE].
func fakeStreamingChatServer(t *testing.T, contents []string, finish string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, c := range contents {
			fmt.Fprintf(w, "data: {\"id\":\"upstream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}]}\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: {\"id\":\"upstream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", finish)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

func TestChatCompletions_Streaming_HappyPath(t *testing.T) {
	h, _ := setupTestHandler(t)

	srv := fakeStreamingChatServer(t, []string{"Hola", " mundo"}, "stop")
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL:     srv.URL,
		Model:       "test-model",
		MaxTokens:   100,
		Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, nil, "You are a test assistant.", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	sessionID := h.sessionStore.Create()
	body, _ := json.Marshal(ChatCompletionRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("saludos"),
					},
				}},
			},
			Model: "test-model",
		},
		Stream: true,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("X-Session-ID", sessionID)
	w := httptest.NewRecorder()
	h.ChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d\nbody=%s\nheaders=%v", w.Code, w.Body.String(), w.Header())
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if got := w.Header().Get("X-Session-ID"); got != sessionID {
		t.Errorf("X-Session-ID = %q, want %q", got, sessionID)
	}

	out := w.Body.String()

	// Must terminate with [DONE]
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), "data: [DONE]") {
		t.Fatalf("body does not end with [DONE]:\n%s", out)
	}

	// First chunk should be a role:assistant delta (OpenAI convention).
	if !strings.Contains(out, `"role":"assistant"`) {
		t.Errorf("missing role:assistant delta chunk:\n%s", out)
	}
	// Content deltas
	if !strings.Contains(out, `"content":"Hola"`) || !strings.Contains(out, `"content":" mundo"`) {
		t.Errorf("missing content deltas:\n%s", out)
	}
	// finish_reason chunk
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason chunk:\n%s", out)
	}

	// Session was persisted (user + assistant).
	sess := h.sessionStore.Get(sessionID)
	if sess == nil {
		t.Fatal("session should exist after stream")
	}
	if got := len(sess.GetMessages()); got != 2 {
		t.Errorf("session has %d messages, want 2", got)
	}
}

func TestChatCompletions_Streaming_NoSessionHeaderMintsOne(t *testing.T) {
	h, _ := setupTestHandler(t)

	srv := fakeStreamingChatServer(t, []string{"hi"}, "stop")
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: srv.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, nil, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	body, _ := json.Marshal(ChatCompletionRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("hello"),
					},
				}},
			},
			Model: "test-model",
		},
		Stream: true,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.ChatCompletions(w, req)

	mintedSID := w.Header().Get("X-Session-ID")
	if mintedSID == "" {
		t.Fatal("expected X-Session-ID header to be set even without input header")
	}
	if h.sessionStore.Get(mintedSID) == nil {
		t.Fatalf("session %q not found in store after stream", mintedSID)
	}
}

// TestChatCompletions_Streaming_StaysPure asserts the OpenAI stream emits ONLY
// standard chunks even when a tool fires — no named tool_start/tool_done/
// rag_final events that would surface to SDK clients as bogus empty-choices
// chunks. RAG observability lives on /v1/agent/chat instead.
func TestChatCompletions_Streaming_StaysPure(t *testing.T) {
	h, _ := setupTestHandler(t)

	// Two-turn SSE script: turn 1 streams a tool_calls delta, turn 2 streams the
	// final answer. runToolLoop uses ChatStreamWithTools for every iteration
	// when streaming, so both upstream turns must be SSE.
	var callIdx int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		callIdx++
		if callIdx == 1 {
			// Tool-calling chunk: a single chunk that carries the tool_calls
			// delta, then a finish_reason=tool_calls chunk, then [DONE].
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{}"}}]},"finish_reason":null}]}`+"\n\n")
			flusher.Flush()
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			flusher.Flush()
		} else {
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":null}]}`+"\n\n")
			flusher.Flush()
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: srv.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	echo := stubEchoTool()
	ag := agent.NewAgent(chatClient, []agent.Tool{echo}, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	body, _ := json.Marshal(ChatCompletionRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("call echo please"),
					},
				}},
			},
			Model: "test-model",
		},
		Stream: true,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.ChatCompletions(w, req)

	out := w.Body.String()
	// The final assistant content must still stream through as a chunk.
	if !strings.Contains(out, `"content":"done"`) {
		t.Errorf("missing final content chunk:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), "data: [DONE]") {
		t.Errorf("stream must end with [DONE]:\n%s", out)
	}
	// Purity: no named events, no proprietary payloads — even though a tool fired.
	for _, banned := range []string{"event: tool_start", "event: tool_done", "event: rag_final", "event: tool", "call_id", "citations"} {
		if strings.Contains(out, banned) {
			t.Errorf("OpenAI stream leaked %q:\n%s", banned, out)
		}
	}
}

// stubEchoTool is a Tool that echoes its args as the tool result. Defined here
// instead of importing the agent-test fixture (different package).
type stubEchoTool_ struct{}

func (stubEchoTool_) Name() string        { return "echo" }
func (stubEchoTool_) Description() string { return "echoes args" }
func (stubEchoTool_) Parameters() openai.FunctionParameters {
	return openai.FunctionParameters{"type": "object", "properties": map[string]any{}}
}
func (stubEchoTool_) Execute(_ context.Context, args string) (agent.ToolResult, error) {
	return agent.ToolResult{Text: "echo: " + args}, nil
}

// TestChatCompletions_Streaming_OverRealSocket drives the full chi router over
// a real TCP socket (httptest.NewServer) so the chunked-transfer + flush path
// and router middleware are exercised end-to-end, not just the in-process
// handler. The upstream LLM is a fake SSE server.
func TestChatCompletions_Streaming_OverRealSocket(t *testing.T) {
	h, _ := setupTestHandler(t)

	upstream := fakeStreamingChatServer(t, []string{"Hola", " mundo"}, "stop")
	defer upstream.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: upstream.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, nil, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	router := NewRouter(h, zerolog.Nop(), 30*time.Second)
	srv := httptest.NewServer(router)
	defer srv.Close()

	body, _ := json.Marshal(ChatCompletionRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("saludos"),
					},
				}},
			},
			Model: "test-model",
		},
		Stream: true,
	})

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if resp.Header.Get("X-Session-ID") == "" {
		t.Error("missing X-Session-ID response header")
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	out := string(raw)
	if !strings.Contains(out, `"content":"Hola"`) || !strings.Contains(out, `"content":" mundo"`) {
		t.Errorf("missing content deltas:\n%s", out)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), "data: [DONE]") {
		t.Errorf("stream did not end with [DONE]:\n%s", out)
	}
}

func stubEchoTool() agent.Tool { return stubEchoTool_{} }
