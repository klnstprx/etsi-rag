package admin

import (
	"net/http"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/agent"
)

// GetAgent returns the currently installed agent (or nil). Thread-safe;
// exposed for diagnostic tests that verify rebuild wiring is preserved.
// The agent itself lives in the shared *agent.Handle owned by the server —
// admin and api both read from that single source of truth.
func (h *Handler) GetAgent() *agent.Agent {
	return h.agentHandle.Get()
}

// ChatPage handles GET /admin/chat. It mints a fresh session and renders the
// chat shell; the browser then talks to the proprietary POST /v1/agent/chat
// endpoint directly (see static/chat.js). The admin UI is just another client
// of that API — it has no chat-specific server endpoints of its own.
func (h *Handler) ChatPage(w http.ResponseWriter, r *http.Request) {
	sessionID := h.sessionStore.Create()

	page := templates.ChatPage(sessionID)
	content := templates.ChatContent(sessionID)
	render(w, r, page, content)
}

// ChatReset handles POST /admin/chat/reset — mints a new session and re-renders
// the (empty) chat shell so the conversation starts fresh.
func (h *Handler) ChatReset(w http.ResponseWriter, r *http.Request) {
	sessionID := h.sessionStore.Create()

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.ChatContent(sessionID).Render(r.Context(), w)
	templates.Toast("Chat reset", "success").Render(r.Context(), w)
}
