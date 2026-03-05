package api

import (
	"encoding/json"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/openai/openai-go"
)

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	mu              sync.RWMutex
	embeddingClient llm.LLM
	chatClient      llm.LLM
	counter         tokencount.Counter
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

// SetCounter swaps the token counter (thread-safe).
func (h *Handler) SetCounter(c tokencount.Counter) {
	h.mu.Lock()
	h.counter = c
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

func (h *Handler) getCounter() tokencount.Counter {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.counter
}

// HandlerConfig holds configuration for creating a Handler.
type HandlerConfig struct {
	EmbeddingClient llm.LLM
	ChatClient      llm.LLM
	Counter         tokencount.Counter
	Pipeline        *rag.Pipeline
	Store           store.Store
	SessionStore    *session.Store
}

// NewHandler creates a new Handler with the given dependencies.
func NewHandler(cfg HandlerConfig) *Handler {
	return &Handler{
		embeddingClient: cfg.EmbeddingClient,
		chatClient:      cfg.ChatClient,
		counter:         cfg.Counter,
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
	json.NewEncoder(w).Encode(openai.ChatCompletion{
		ID:      "chatcmpl-local",
		Model:   string(req.Model),
		Created: time.Now().Unix(),
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Content: resp,
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
			ID:      "ragchat-local",
			Model:   req.Model,
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
