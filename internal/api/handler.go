package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
)

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	agentHandle  *agent.Handle
	retriever    *rag.Retriever
	ingester     *ingest.Ingester
	store        *store.SQLiteStore
	sessionStore *session.Store
	objStore     *objectstore.Client
}

// HandlerConfig holds configuration for creating a Handler.
//
// The agent and embedder live on shared handles owned by the server; api and
// admin both read from those single sources of truth instead of holding
// their own copies that must be hot-swapped in parallel.
type HandlerConfig struct {
	AgentHandle  *agent.Handle
	Retriever    *rag.Retriever
	Ingester     *ingest.Ingester
	Store        *store.SQLiteStore
	SessionStore *session.Store
	ObjStore     *objectstore.Client
}

// NewHandler creates a new Handler with the given dependencies.
func NewHandler(cfg HandlerConfig) *Handler {
	return &Handler{
		agentHandle:  cfg.AgentHandle,
		retriever:    cfg.Retriever,
		ingester:     cfg.Ingester,
		store:        cfg.Store,
		sessionStore: cfg.SessionStore,
		objStore:     cfg.ObjStore,
	}
}

// HealthCheck handles GET /health.
func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
}

// ChatCompletions handles POST /v1/chat/completions. Single user message,
// session managed via X-Session-ID header. When req.Stream is set it emits a
// strictly OpenAI-compatible SSE stream; otherwise a single openai.ChatCompletion.
//
// The response is the standard OpenAI shape and nothing more. The agent
// resolves all tool calls server-side, so the standard choices[].message.
// tool_calls is always null here. The proprietary RAG context — citations,
// images, and the root-level tool-call audit trail — is NOT exposed on this
// endpoint; clients that want it use POST /v1/agent/chat.
func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	ag := h.agentHandle.Get()
	if ag == nil {
		ErrServiceUnavailable("agent not configured").WriteJSON(w, r)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w, r)
		return
	}

	// Validate: exactly 1 user message, no system/assistant
	userMsg, err := extractSingleUserMessage(req.Messages)
	if err != nil {
		ErrBadRequest(err.Error()).WriteJSON(w, r)
		return
	}

	// Get or create session
	sessionIDHeader := r.Header.Get("X-Session-ID")
	var sessionIDPtr *string
	if sessionIDHeader != "" {
		sessionIDPtr = &sessionIDHeader
	}
	sessionID, _ := h.sessionStore.GetOrCreate(sessionIDPtr)

	if req.Stream {
		h.chatSessionStreaming(w, r, ag, sessionID, string(req.Model), userMsg)
		return
	}

	// Run agent
	result, err := ag.Chat(r.Context(), sessionID, userMsg, nil)
	if err != nil {
		classifyUpstreamError(err).WriteJSON(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Session-ID", sessionID)
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

// chatSessionStreaming runs the agent through agent.RelayChat and writes the
// resulting tokens / tool events / outcome to the response as OpenAI-spec SSE.
// X-Session-ID is set before the first chunk goes out so the client can
// persist it without parsing the body.
func (h *Handler) chatSessionStreaming(w http.ResponseWriter, r *http.Request, ag *agent.Agent, sessionID, model, userMsg string) {
	flusher, err := prepareSSE(w)
	if err != nil {
		ErrInternal(err).WriteJSON(w, r)
		return
	}
	w.Header().Set("X-Session-ID", sessionID)

	sink := newOpenAISSESink(w, flusher, model)
	// Errors from RelayChat almost always mean the client disconnected mid-stream
	// (writes to w fail). We can't change the status code after headers have been
	// flushed, so just log at debug for observability.
	if err := agent.RelayChat(r.Context(), ag, sessionID, userMsg, sink); err != nil {
		logger.Debug().
			Err(err).
			Str("path", r.URL.Path).
			Str("request_id", middleware.GetReqID(r.Context())).
			Msg("chat stream relay ended early (likely client disconnect)")
	}
}

// Ingest handles POST /v1/rag/ingest.
func (h *Handler) Ingest(w http.ResponseWriter, r *http.Request) {
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w, r)
		return
	}

	if len(req.Documents) == 0 {
		ErrBadRequest("documents cannot be empty").WriteJSON(w, r)
		return
	}

	// Convert to domain documents
	docs := make([]domain.Document, len(req.Documents))
	for i, d := range req.Documents {
		lang := domain.Language(d.Language)
		if !lang.Valid() {
			ErrInvalidLanguage(d.Language).WriteJSON(w, r)
			return
		}
		docs[i] = domain.Document{
			ID:       d.ID,
			Title:    d.Title,
			Content:  d.Content,
			URL:      d.URL,
			Language: lang,
		}
	}

	ingested, chunks, err := h.ingester.Ingest(r.Context(), docs)
	if err != nil {
		classifyUpstreamError(err).WriteJSON(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(IngestResponse{
		IngestedCount: ingested,
		ChunksCreated: chunks,
	})
}

// RAGQuery handles POST /v1/rag/query — retrieval-layer introspection.
//
// Returns the top-k chunks the retrieval layer would surface for a query,
// without going through the agent's tool-calling loop. Used by the eval
// harness to compute Recall@k / MRR against a ground-truth golden URL set.
// The response intentionally exposes document_url so the caller can map
// chunks back to their source document.
func (h *Handler) RAGQuery(w http.ResponseWriter, r *http.Request) {
	var req RAGQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w, r)
		return
	}
	if req.Query == "" {
		ErrBadRequest("query cannot be empty").WriteJSON(w, r)
		return
	}

	result, err := h.retriever.Retrieve(r.Context(), req.Query, rag.RetrieveOpts{TopK: req.TopK})
	if err != nil {
		classifyUpstreamError(err).WriteJSON(w, r)
		return
	}

	docByID := make(map[string]domain.Document, len(result.Documents))
	for _, d := range result.Documents {
		docByID[d.ID] = d
	}

	out := make([]RetrievedChunk, 0, len(result.Chunks))
	for _, sc := range result.Chunks {
		d := docByID[sc.Chunk.DocID]
		out = append(out, RetrievedChunk{
			ChunkID:       sc.Chunk.ID,
			DocumentID:    d.ID,
			DocumentTitle: d.Title,
			DocumentURL:   d.URL,
			SectionTitle:  sc.Chunk.SectionTitle,
			ChunkIndex:    sc.Chunk.ChunkIndex,
			Score:         sc.Score,
			Snippet:       sc.Chunk.Content,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(RAGQueryResponse{Results: out})
}

// ListDocuments handles GET /v1/rag/documents.
func (h *Handler) ListDocuments(w http.ResponseWriter, r *http.Request) {
	var lang *domain.Language
	if l := r.URL.Query().Get("language"); l != "" {
		dl := domain.Language(l)
		if !dl.Valid() {
			ErrInvalidLanguage(l).WriteJSON(w, r)
			return
		}
		lang = &dl
	}

	docs, err := h.store.ListDocuments(r.Context(), lang)
	if err != nil {
		ErrInternal(err).WriteJSON(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(docs)
}

// GetArtifact handles GET /v1/artifacts/{token}.png. The {token} path
// component is the artifact's opaque public_token (random 128-bit hex), not
// its raw integer ID — this prevents trivial enumeration of all stored
// artifact images. The handler streams the underlying object from RustFS
// with the artifact's stored MIME type.
func (h *Handler) GetArtifact(w http.ResponseWriter, r *http.Request) {
	if h.objStore == nil {
		ErrInternal(errors.New("object store not configured")).WriteJSON(w, r)
		return
	}
	token := chi.URLParam(r, "token")
	// Strip a trailing ".png" extension if present so /v1/artifacts/abc and
	// /v1/artifacts/abc.png both resolve to the same row. Storing the
	// artifact's MIME type means we don't actually rely on the extension.
	if dot := len(token) - 4; dot > 0 && token[dot:] == ".png" {
		token = token[:dot]
	}
	if token == "" {
		ErrBadRequest("missing artifact token").WriteJSON(w, r)
		return
	}

	a, err := h.store.GetArtifactByToken(r.Context(), token)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			ErrNotFound("artifact not found").WriteJSON(w, r)
			return
		}
		ErrInternal(err).WriteJSON(w, r)
		return
	}

	data, err := h.objStore.Get(r.Context(), a.ObjectStoreKey)
	if err != nil {
		ErrInternal(fmt.Errorf("fetching artifact bytes: %w", err)).WriteJSON(w, r)
		return
	}

	mime := a.MimeType
	if mime == "" {
		mime = "image/png"
	}
	w.Header().Set("Content-Type", mime)
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Write(data)
}

// DeleteDocument handles DELETE /v1/rag/documents/{id}.
func (h *Handler) DeleteDocument(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		ErrBadRequest("missing document id").WriteJSON(w, r)
		return
	}

	err := h.store.DeleteDocument(r.Context(), id)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			ErrNotFound("document not found").WriteJSON(w, r)
			return
		}
		ErrInternal(err).WriteJSON(w, r)
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
