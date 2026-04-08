package admin

import (
	"context"
	"fmt"
	"html"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/agent"
)

// chatStream holds state for an active admin chat session.
// The agent goroutine writes tokens; ChatStream reads them via SSE.
type chatStream struct {
	tokens chan string
	cancel context.CancelFunc
}

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

	// Create a cancellable context for the agent goroutine.
	// Ownership: created here, canceled by ChatStream (on SSE disconnect or completion).
	// Timeout is a safety net for orphaned goroutines (SSE never connects).
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	tokenChan := make(chan string, 64)
	stream := &chatStream{tokens: tokenChan, cancel: cancel}

	h.chatMu.Lock()
	h.chatStreams[sessionID] = stream
	h.chatMu.Unlock()

	// Run agent in background if available
	if h.agent != nil {
		go func() {
			defer func() {
				cancel()
				close(tokenChan)
				h.chatMu.Lock()
				delete(h.chatStreams, sessionID)
				h.chatMu.Unlock()
			}()

			result, err := h.agent.Chat(ctx, sessionID, message)
			if err != nil {
				tokenChan <- fmt.Sprintf("[Error: %s]", err.Error())
				return
			}
			tokenChan <- result.Response
		}()
	} else {
		// No agent configured, send placeholder
		go func() {
			defer func() {
				cancel()
				close(tokenChan)
				h.chatMu.Lock()
				delete(h.chatStreams, sessionID)
				h.chatMu.Unlock()
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

	h.chatMu.RLock()
	stream, ok := h.chatStreams[sessionID]
	h.chatMu.RUnlock()

	if !ok {
		http.Error(w, "stream not found", http.StatusNotFound)
		return
	}

	// Cancel agent context when SSE exits (client disconnect or natural completion).
	defer stream.cancel()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case token, open := <-stream.tokens:
			if !open {
				// Channel closed — agent finished.
				fmt.Fprintf(w, "event: done\ndata: \n\n")
				flusher.Flush()
				return
			}
			fmt.Fprintf(w, "event: token\ndata: %s\n\n", html.EscapeString(token))
			flusher.Flush()
		case <-r.Context().Done():
			// SSE client disconnected — defer cancels the agent.
			return
		}
	}
}

// ChatReset handles POST /admin/chat/reset.
func (h *Handler) ChatReset(w http.ResponseWriter, r *http.Request) {
	sessionID := h.sessionStore.Create()

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.ChatContent(sessionID).Render(r.Context(), w)
	templates.Toast("Chat reset", "success").Render(r.Context(), w)
}
