package admin

import (
	"net/http"

	"etsi-rag/internal/admin/templates"
)

// Dashboard handles GET /admin/.
func (h *Handler) Dashboard(w http.ResponseWriter, r *http.Request) {
	docs, err := h.store.ListDocuments(r.Context(), nil)
	if err != nil {
		h.logger.Error().Err(err).Msg("listing documents")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	data := templates.DashboardData{
		DocCount:     len(docs),
		BackendCount: len(h.config.Backends),
		Backends:     h.config.Backends,
	}

	page := templates.DashboardPage(data)
	content := templates.DashboardContent(data)
	render(w, r, page, content)
}
