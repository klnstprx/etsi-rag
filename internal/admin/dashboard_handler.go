package admin

import (
	"net/http"

	"etsi-rag/internal/admin/templates"
)

// Dashboard handles GET /admin/.
func (h *Handler) Dashboard(w http.ResponseWriter, r *http.Request) {
	docCount, err := h.store.CountDocuments(r.Context())
	if err != nil {
		h.logger.Error().Err(err).Msg("counting documents")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	chunkCount, err := h.store.CountChunks(r.Context())
	if err != nil {
		h.logger.Error().Err(err).Msg("counting chunks")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	data := templates.DashboardData{
		DocCount:     docCount,
		ChunkCount:   chunkCount,
		BackendCount: len(h.config.Backends),
		Backends:     h.config.Backends,
		HasObjStore:  h.objStore != nil,
	}

	page := templates.DashboardPage(data)
	content := templates.DashboardContent(data)
	render(w, r, page, content)
}

// HealthObjectStore handles GET /admin/health/objectstore — returns health badge for RustFS.
func (h *Handler) HealthObjectStore(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")

	healthy := false
	if h.objStore != nil {
		healthy = h.objStore.Healthy(r.Context())
	}

	status := templates.ServiceStatus{
		Name:    "Object Store",
		URL:     h.config.ObjectStore.Endpoint,
		Healthy: healthy,
	}

	templates.HealthGrid([]templates.ServiceStatus{status}).Render(r.Context(), w)
}
