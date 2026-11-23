package admin

import (
	"net/http"

	"etsi-rag/internal/admin/templates"
)

// ChatPage handles GET /admin/chat. It mints a fresh session and renders the
// chat shell; the browser then talks to POST /v1/agent/chat directly (see
// static/chat.js), so admin has no chat endpoints of its own.
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

	writeHTML(w, r, templates.ChatContent(sessionID), templates.Toast("Chat reset", "success"))
}
