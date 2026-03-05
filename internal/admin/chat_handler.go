package admin

import (
	"fmt"
	"html"
	"net/http"
	"sync"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/agent"
)

// chatStreams holds active SSE channels keyed by session ID.
var chatStreams = struct {
	mu      sync.RWMutex
	streams map[string]chan string
}{streams: make(map[string]chan string)}

// SetAgent sets the agent instance for chat handlers.
func (h *Handler) SetAgent(a *agent.Agent) {
	h.agent = a
}

// ChatPage handles GET /admin/chat.
func (h *Handler) ChatPage(w http.ResponseWriter, r *http.Request) {
	sessionID := h.sessionStore.Create()

	page := templates.ChatPage(sessionID)
	content := templates.ChatContent(sessionID)
	render(w, r, page, content)
}

// ChatSend handles POST /admin/chat/send.
func (h *Handler) ChatSend(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	message := r.FormValue("message")
	sessionID := r.FormValue("session_id")

	if message == "" || sessionID == "" {
		http.Error(w, "missing message or session_id", http.StatusBadRequest)
		return
	}

	// Create a stream channel for this session
	tokenChan := make(chan string, 64)
	chatStreams.mu.Lock()
	chatStreams.streams[sessionID] = tokenChan
	chatStreams.mu.Unlock()

	// Run agent in background if available
	if h.agent != nil {
		go func() {
			defer func() {
				close(tokenChan)
				chatStreams.mu.Lock()
				delete(chatStreams.streams, sessionID)
				chatStreams.mu.Unlock()
			}()

			result, err := h.agent.Run(r.Context(), message)
			if err != nil {
				tokenChan <- fmt.Sprintf("[Error: %s]", err.Error())
				return
			}
			// Send full response as single token (agent returns complete response)
			tokenChan <- result
		}()
	} else {
		// No agent configured, send placeholder
		go func() {
			defer func() {
				close(tokenChan)
				chatStreams.mu.Lock()
				delete(chatStreams.streams, sessionID)
				chatStreams.mu.Unlock()
			}()
			tokenChan <- "Agent not configured. Please configure the [agent] section in config."
		}()
	}

	// Return user message + SSE div
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.UserMessage(message).Render(r.Context(), w)
	templates.AssistantMessageSSE(sessionID).Render(r.Context(), w)
}

// ChatStream handles GET /admin/chat/stream/{sessionID} (SSE endpoint).
func (h *Handler) ChatStream(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "sessionID")

	chatStreams.mu.RLock()
	tokenChan, ok := chatStreams.streams[sessionID]
	chatStreams.mu.RUnlock()

	if !ok {
		http.Error(w, "stream not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	for token := range tokenChan {
		fmt.Fprintf(w, "event: token\ndata: %s\n\n", html.EscapeString(token))
		flusher.Flush()
	}

	fmt.Fprintf(w, "event: done\ndata: \n\n")
	flusher.Flush()
}

// ChatReset handles POST /admin/chat/reset.
func (h *Handler) ChatReset(w http.ResponseWriter, r *http.Request) {
	sessionID := h.sessionStore.Create()

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.ChatContent(sessionID).Render(r.Context(), w)
	templates.Toast("Chat reset", "success").Render(r.Context(), w)
}
