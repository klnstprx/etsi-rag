package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
)

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	mu              sync.RWMutex
	embeddingClient llm.Embedder
	counter         tokencount.Counter
	agent           *agent.Agent
	pipeline        *rag.Pipeline
	store           *store.SQLiteStore
	sessionStore    *session.Store
	passthrough     bool
}

// SetEmbeddingClient swaps the embedding client (thread-safe).
func (h *Handler) SetEmbeddingClient(c llm.Embedder) {
	h.mu.Lock()
	h.embeddingClient = c
	h.mu.Unlock()
}

// SetCounter swaps the token counter (thread-safe).
func (h *Handler) SetCounter(c tokencount.Counter) {
	h.mu.Lock()
	h.counter = c
	h.mu.Unlock()
}

// SetAgent swaps the agent (thread-safe).
func (h *Handler) SetAgent(a *agent.Agent) {
	h.mu.Lock()
	h.agent = a
	h.mu.Unlock()
}

// SetPassthrough toggles passthrough mode (thread-safe).
func (h *Handler) SetPassthrough(p bool) {
	h.mu.Lock()
	h.passthrough = p
	h.mu.Unlock()
}

func (h *Handler) getEmbeddingClient() llm.Embedder {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.embeddingClient
}

func (h *Handler) getCounter() tokencount.Counter {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.counter
}

func (h *Handler) getAgent() *agent.Agent {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.agent
}

func (h *Handler) getPassthrough() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.passthrough
}

// HandlerConfig holds configuration for creating a Handler.
type HandlerConfig struct {
	EmbeddingClient llm.Embedder
	Counter         tokencount.Counter
	Agent           *agent.Agent
	Pipeline        *rag.Pipeline
	Store           *store.SQLiteStore
	SessionStore    *session.Store
	Passthrough     bool
}

// NewHandler creates a new Handler with the given dependencies.
func NewHandler(cfg HandlerConfig) *Handler {
	return &Handler{
		embeddingClient: cfg.EmbeddingClient,
		counter:         cfg.Counter,
		agent:           cfg.Agent,
		pipeline:        cfg.Pipeline,
		store:           cfg.Store,
		sessionStore:    cfg.SessionStore,
		passthrough:     cfg.Passthrough,
	}
}

// HealthCheck handles GET /health.
func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

// ChatCompletions handles POST /v1/chat/completions.
// Two modes based on passthrough config:
// - Session mode (default): single user message, session managed via X-Session-ID header
// - Passthrough mode: accepts full message array, no session management
func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	ag := h.getAgent()
	if ag == nil {
		ErrServiceUnavailable("agent not configured").WriteJSON(w)
		return
	}

	if h.getPassthrough() {
		h.chatPassthrough(w, r, ag)
	} else {
		h.chatSession(w, r, ag)
	}
}

// chatSession handles session-managed chat.
func (h *Handler) chatSession(w http.ResponseWriter, r *http.Request, ag *agent.Agent) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	// Validate: exactly 1 user message, no system/assistant
	userMsg, err := extractSingleUserMessage(req.Messages)
	if err != nil {
		ErrBadRequest(err.Error()).WriteJSON(w)
		return
	}

	// Get or create session
	sessionIDHeader := r.Header.Get("X-Session-ID")
	var sessionIDPtr *string
	if sessionIDHeader != "" {
		sessionIDPtr = &sessionIDHeader
	}
	sessionID, _ := h.sessionStore.GetOrCreate(sessionIDPtr)

	// Run agent
	result, err := ag.Chat(r.Context(), sessionID, userMsg)
	if err != nil {
		ErrInternal(err).WriteJSON(w)
		return
	}

	// Build response
	resp := ChatCompletionWithSources{
		ChatCompletion: openai.ChatCompletion{
			ID:      "chatcmpl-local",
			Model:   string(req.Model),
			Created: time.Now().Unix(),
			Choices: []openai.ChatCompletionChoice{
				{
					Index:        0,
					FinishReason: "stop",
					Message: openai.ChatCompletionMessage{
						Content: result.Response,
					},
				},
			},
		},
		Sources: result.Sources,
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Session-ID", sessionID)
	json.NewEncoder(w).Encode(resp)
}

// chatPassthrough handles stateless passthrough chat.
func (h *Handler) chatPassthrough(w http.ResponseWriter, r *http.Request, ag *agent.Agent) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	if len(req.Messages) == 0 {
		ErrBadRequest("messages cannot be empty").WriteJSON(w)
		return
	}

	result, err := ag.ChatPassthrough(r.Context(), req.Messages)
	if err != nil {
		ErrInternal(err).WriteJSON(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openai.ChatCompletion{
		ID:      "chatcmpl-local",
		Model:   string(req.Model),
		Created: time.Now().Unix(),
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Content: result.Response,
				},
			},
		},
	})
}

// Embeddings handles POST /v1/embeddings.
func (h *Handler) Embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w)
		return
	}

	// Extract texts from the input union.
	var texts []string
	if req.Input.OfString.Valid() {
		texts = []string{req.Input.OfString.Value}
	} else if len(req.Input.OfArrayOfStrings) > 0 {
		texts = req.Input.OfArrayOfStrings
	} else {
		ErrBadRequest("input must be a string or array of strings").WriteJSON(w)
		return
	}

	embeddings, err := h.getEmbeddingClient().EmbedBatch(r.Context(), texts)
	if err != nil {
		ErrBadGateway(err).WriteJSON(w)
		return
	}

	// Build response using SDK types.
	data := make([]openai.Embedding, len(embeddings))
	var totalTokens int64
	counter := h.getCounter()
	for i, emb := range embeddings {
		emb64 := make([]float64, len(emb))
		for j, v := range emb {
			emb64[j] = float64(v)
		}
		data[i] = openai.Embedding{
			Index:     int64(i),
			Embedding: emb64,
		}
		if counter != nil {
			n, err := counter.Count(r.Context(), texts[i])
			if err != nil {
				ErrInternal(err).WriteJSON(w)
				return
			}
			totalTokens += int64(n)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openai.CreateEmbeddingResponse{
		Model: req.Model,
		Data:  data,
		Usage: openai.CreateEmbeddingResponseUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	})
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
		if errors.Is(err, store.ErrNotFound) {
			ErrNotFound("document not found").WriteJSON(w)
			return
		}
		ErrInternal(err).WriteJSON(w)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// extractSingleUserMessage validates that messages contain exactly 1 user message
// with no system/assistant messages (for session mode).
func extractSingleUserMessage(messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("messages cannot be empty")
	}

	var userContent string
	userCount := 0

	for _, msg := range messages {
		if msg.OfSystem != nil {
			return "", fmt.Errorf("system messages not allowed in session mode")
		}
		if msg.OfAssistant != nil {
			return "", fmt.Errorf("assistant messages not allowed in session mode")
		}
		if msg.OfTool != nil {
			return "", fmt.Errorf("tool messages not allowed in session mode")
		}
		if msg.OfUser != nil {
			userCount++
			if msg.OfUser.Content.OfString.Valid() {
				userContent = msg.OfUser.Content.OfString.Value
			}
		}
	}

	if userCount != 1 {
		return "", fmt.Errorf("session mode requires exactly 1 user message, got %d", userCount)
	}
	if userContent == "" {
		return "", fmt.Errorf("user message content cannot be empty")
	}

	return userContent, nil
}
