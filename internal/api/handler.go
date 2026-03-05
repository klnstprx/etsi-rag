package api

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
)

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	mu              sync.RWMutex
	embeddingClient llm.LLM
	chatClient      llm.LLM
	pipeline        *rag.Pipeline
	store           store.Store
	sessionStore    *session.Store
}

// SetEmbeddingClient swaps the embedding client (thread-safe).
func (h *Handler) SetEmbeddingClient(c llm.LLM) {
	h.mu.Lock()
	h.embeddingClient = c
	h.mu.Unlock()
}

// SetChatClient swaps the chat client (thread-safe).
func (h *Handler) SetChatClient(c llm.LLM) {
	h.mu.Lock()
	h.chatClient = c
	h.mu.Unlock()
}

func (h *Handler) getEmbeddingClient() llm.LLM {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.embeddingClient
}

func (h *Handler) getChatClient() llm.LLM {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.chatClient
}

// HandlerConfig holds configuration for creating a Handler.
type HandlerConfig struct {
	EmbeddingClient llm.LLM
	ChatClient      llm.LLM
	Pipeline        *rag.Pipeline
	Store           store.Store
	SessionStore    *session.Store
}

// NewHandler creates a new Handler with the given dependencies.
func NewHandler(cfg HandlerConfig) *Handler {
	return &Handler{
		embeddingClient: cfg.EmbeddingClient,
		chatClient:      cfg.ChatClient,
		pipeline:        cfg.Pipeline,
		store:           cfg.Store,
		sessionStore:    cfg.SessionStore,
	}
}

// HealthCheck handles GET /health.
func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

// ChatCompletions handles POST /v1/chat/completions.
func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	resp, err := h.getChatClient().Chat(r.Context(), req.Messages)
	if err != nil {
		ErrBadGateway(err).WriteJSON(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":     "chatcmpl-local",
		"object": "chat.completion",
		"model":  string(req.Model),
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]string{
					"role":    "assistant",
					"content": resp,
				},
			},
		},
	})
}

// Embeddings handles POST /v1/embeddings.
func (h *Handler) Embeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Input any                   `json:"input"`
		Model openai.EmbeddingModel `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	// Handle single string input
	var text string
	switch v := req.Input.(type) {
	case string:
		text = v
	default:
		ErrBadRequest("input must be a string").WriteJSON(w)
		return
	}

	emb, err := h.getEmbeddingClient().Embed(r.Context(), text)
	if err != nil {
		ErrBadGateway(err).WriteJSON(w)
		return
	}

	// Convert to float64 for JSON
	emb64 := make([]float64, len(emb))
	for i, v := range emb {
		emb64[i] = float64(v)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"model":  string(req.Model),
		"data": []map[string]any{
			{
				"object":    "embedding",
				"index":     0,
				"embedding": emb64,
			},
		},
		"usage": map[string]int{
			"prompt_tokens": len(text) / 4,
			"total_tokens":  len(text) / 4,
		},
	})
}

// RAGChat handles POST /v1/rag/chat.
func (h *Handler) RAGChat(w http.ResponseWriter, r *http.Request) {
	var req RAGChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	if len(req.Messages) == 0 {
		ErrBadRequest("messages cannot be empty").WriteJSON(w)
		return
	}

	// Validate language if provided
	lang := "es" // default
	if req.Language != nil {
		l := domain.Language(*req.Language)
		if !l.Valid() {
			ErrInvalidLanguage(*req.Language).WriteJSON(w)
			return
		}
		lang = *req.Language
	}

	result, err := h.pipeline.Chat(r.Context(), req.Messages, lang, req.TopK)
	if err != nil {
		ErrInternal(err).WriteJSON(w)
		return
	}

	resp := RAGChatResponse{
		ChatCompletion: openai.ChatCompletion{
			ID:    "ragchat-local",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index:        0,
					FinishReason: "stop",
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: result.Response,
					},
				},
			},
		},
	}

	if req.IncludeSources {
		for _, src := range result.Sources {
			resp.Sources = append(resp.Sources, SourceReference{
				ID:         src.ID,
				Title:      src.Title,
				URL:        src.URL,
				ChunkIndex: src.ChunkIndex,
				Score:      src.Score,
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Ingest handles POST /v1/rag/ingest.
func (h *Handler) Ingest(w http.ResponseWriter, r *http.Request) {
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	if len(req.Documents) == 0 {
		ErrBadRequest("documents cannot be empty").WriteJSON(w)
		return
	}

	// Convert to domain documents
	docs := make([]domain.Document, len(req.Documents))
	for i, d := range req.Documents {
		lang := domain.Language(d.Language)
		if !lang.Valid() {
			ErrInvalidLanguage(d.Language).WriteJSON(w)
			return
		}
		cat := domain.Category(d.Category)
		if !cat.Valid() {
			ErrBadRequest("invalid category: " + d.Category).WriteJSON(w)
			return
		}
		docs[i] = domain.Document{
			ID:       d.ID,
			Title:    d.Title,
			Content:  d.Content,
			URL:      d.URL,
			Language: lang,
			Category: cat,
		}
	}

	ingested, chunks, err := h.pipeline.Ingest(r.Context(), docs)
	if err != nil {
		ErrInternal(err).WriteJSON(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(IngestResponse{
		IngestedCount: ingested,
		ChunksCreated: chunks,
	})
}

// ListDocuments handles GET /v1/rag/documents.
func (h *Handler) ListDocuments(w http.ResponseWriter, r *http.Request) {
	var lang *domain.Language
	if l := r.URL.Query().Get("language"); l != "" {
		dl := domain.Language(l)
		if !dl.Valid() {
			ErrInvalidLanguage(l).WriteJSON(w)
			return
		}
		lang = &dl
	}

	docs, err := h.store.ListDocuments(r.Context(), lang)
	if err != nil {
		ErrInternal(err).WriteJSON(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(docs)
}

// DeleteDocument handles DELETE /v1/rag/documents/{id}.
func (h *Handler) DeleteDocument(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		ErrBadRequest("missing document id").WriteJSON(w)
		return
	}

	err := h.store.DeleteDocument(r.Context(), id)
	if err != nil {
		// Check if not found
		if err.Error() == "document not found: "+id {
			ErrNotFound("document not found").WriteJSON(w)
			return
		}
		ErrInternal(err).WriteJSON(w)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
