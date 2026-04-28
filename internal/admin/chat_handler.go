package admin

import (
	"bytes"
	"context"
	"fmt"
	"html"
	"io"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/agent"
)

// chatStream holds state for an active admin chat session.
// The agent goroutine writes tokens and tool events; ChatStream reads them via SSE.
type chatStream struct {
	tokens chan string
	events chan agent.ToolEvent
	cancel context.CancelFunc
}

// SetAgent sets the agent instance for chat handlers (thread-safe).
func (h *Handler) SetAgent(a *agent.Agent) {
	h.agent.Store(a)
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
	streamMode := r.FormValue("stream") != ""

	if message == "" || sessionID == "" {
		http.Error(w, "missing message or session_id", http.StatusBadRequest)
		return
	}

	// Create a cancellable context for the agent goroutine.
	// Ownership: created here, canceled by ChatStream (on SSE disconnect or completion).
	// Timeout is a safety net for orphaned goroutines (SSE never connects).
	ctx, cancel := context.WithTimeout(context.Background(), h.config.Admin.ChatStreamTimeout)
	tokenChan := make(chan string, 64)
	eventChan := make(chan agent.ToolEvent, 16)
	stream := &chatStream{tokens: tokenChan, events: eventChan, cancel: cancel}

	h.chatStreams.register(sessionID, stream)

	// Run agent in background if available.
	// The goroutine only closes the channels when done — ChatStream handles
	// map cleanup, so the stream entry survives until the SSE client connects.
	ag := h.agent.Load()
	if ag == nil {
		go func() {
			defer close(tokenChan)
			defer close(eventChan)
			tokenChan <- "Agent not configured. Please configure the [agent] section in config."
		}()
	} else if streamMode {
		go func() {
			defer close(tokenChan)
			defer close(eventChan)
			tokens, outcome := ag.ChatStream(ctx, sessionID, message, eventChan)
			for tok := range tokens {
				tokenChan <- tok
			}
			if out := <-outcome; out.Err != nil {
				tokenChan <- fmt.Sprintf("[Error: %s]", out.Err.Error())
			}
		}()
	} else {
		go func() {
			defer close(tokenChan)
			defer close(eventChan)
			result, err := ag.Chat(ctx, sessionID, message, eventChan)
			if err != nil {
				tokenChan <- fmt.Sprintf("[Error: %s]", err.Error())
				return
			}
			tokenChan <- result.Response
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

	stream, ok := h.chatStreams.get(sessionID)

	if !ok {
		http.Error(w, "stream not found", http.StatusNotFound)
		return
	}

	// Clean up when SSE exits (client disconnect or natural completion).
	defer func() {
		stream.cancel()
		h.chatStreams.remove(sessionID)
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	events := stream.events
	tokens := stream.tokens

	for {
		select {
		case evt, open := <-events:
			if !open {
				events = nil
			} else {
				renderToolEvent(w, flusher, evt)
			}
		case token, open := <-tokens:
			if !open {
				tokens = nil
			} else {
				writeSSE(w, "token", html.EscapeString(token))
				flusher.Flush()
			}
		case <-r.Context().Done():
			return
		}

		if events == nil && tokens == nil {
			writeSSE(w, "done", "")
			flusher.Flush()
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

// writeSSE writes a properly formatted SSE event with multi-line data support.
func writeSSE(w io.Writer, event, data string) {
	fmt.Fprintf(w, "event: %s\n", event)
	for line := range strings.SplitSeq(data, "\n") {
		fmt.Fprintf(w, "data: %s\n", line)
	}
	fmt.Fprint(w, "\n")
}

// renderToolEvent renders a tool event as an SSE event with HTML data.
func renderToolEvent(w http.ResponseWriter, flusher http.Flusher, evt agent.ToolEvent) {
	var buf bytes.Buffer
	switch evt.Type {
	case "start":
		templates.ToolCardLoading(evt.CallID, evt.Name).Render(context.Background(), &buf)
		writeSSE(w, "tool_start", buf.String())
	case "done":
		templates.ToolCardDone(evt.CallID, evt.Name, evt.Duration, evt.Error, evt.Citations).Render(context.Background(), &buf)
		writeSSE(w, "tool_done", buf.String())
	}
	flusher.Flush()
}
