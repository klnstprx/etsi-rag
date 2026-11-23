package admin

import (
	"context"
	"net/http"
	"slices"
	"time"

	"github.com/a-h/templ"
	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/backend"
	"etsi-rag/internal/config"
)

// BackendGetRow handles GET /admin/backends/{name} — returns the display row (used by Cancel).
func (h *Handler) BackendGetRow(w http.ResponseWriter, r *http.Request) {
	h.writeBackendRow(w, r, templates.BackendRow)
}

// BackendEditForm handles GET /admin/backends/{name}/edit — returns an inline edit row.
func (h *Handler) BackendEditForm(w http.ResponseWriter, r *http.Request) {
	h.writeBackendRow(w, r, templates.BackendEditRow)
}

// writeBackendRow renders row for the enabled backend named in the URL, or 404s.
func (h *Handler) writeBackendRow(w http.ResponseWriter, r *http.Request, row func(config.BackendConfig) templ.Component) {
	b := h.config.EnabledBackend(chi.URLParam(r, "name"))
	if b == nil {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}
	writeHTML(w, r, row(*b))
}

// BackendList handles GET /admin/backends.
func (h *Handler) BackendList(w http.ResponseWriter, r *http.Request) {
	page := templates.BackendsPage(h.config.Backends)
	content := templates.BackendsContent(h.config.Backends)
	render(w, r, page, content)
}

// BackendCreate handles POST /admin/backends.
func (h *Handler) BackendCreate(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	b := config.BackendConfig{
		Name:       r.FormValue("name"),
		BaseURL:    r.FormValue("base_url"),
		APIKey:     r.FormValue("api_key"),
		Enabled:    r.FormValue("enabled") == "true",
		IsLlamaCpp: r.FormValue("is_llama_cpp") == "true",
	}

	h.config.Backends = append(h.config.Backends, b)

	if err := h.saveConfig(); err != nil {
		// Roll back
		h.config.Backends = h.config.Backends[:len(h.config.Backends)-1]
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.logger.Info().Str("name", b.Name).Msg("backend created")
	h.renderBackendList(w, r, "Backend created")
}

// BackendUpdate handles PUT /admin/backends/{name}.
func (h *Handler) BackendUpdate(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")

	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	idx := slices.IndexFunc(h.config.Backends, func(b config.BackendConfig) bool { return b.Name == name })
	if idx < 0 {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}

	h.config.Backends[idx].Name = r.FormValue("name")
	h.config.Backends[idx].BaseURL = r.FormValue("base_url")
	h.config.Backends[idx].Enabled = r.FormValue("enabled") == "true"
	h.config.Backends[idx].IsLlamaCpp = r.FormValue("is_llama_cpp") == "true"

	// Only overwrite API key if a new value was provided
	if key := r.FormValue("api_key"); key != "" {
		h.config.Backends[idx].APIKey = key
	}

	if err := h.saveConfig(); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.renderBackendList(w, r, "Backend updated")
}

// BackendDelete handles DELETE /admin/backends/{name}.
func (h *Handler) BackendDelete(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")

	idx := slices.IndexFunc(h.config.Backends, func(b config.BackendConfig) bool { return b.Name == name })
	if idx < 0 {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}

	h.config.Backends = append(h.config.Backends[:idx], h.config.Backends[idx+1:]...)

	if err := h.saveConfig(); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.logger.Info().Str("name", name).Msg("backend deleted")
	h.renderBackendList(w, r, "Backend deleted")
}

// BackendHealth handles GET /admin/backends/{name}/health.
func (h *Handler) BackendHealth(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")

	b := h.config.EnabledBackend(name)
	healthy := b != nil && checkBackendHealth(b.BaseURL, h.config.Admin.BackendProbeTimeout)
	writeHTML(w, r, templates.StatusBadge(healthy))
}

// BackendModels handles GET /admin/models?backend=X&current=Y — returns <option> elements.
func (h *Handler) BackendModels(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("backend")
	selected := r.URL.Query().Get("current")

	var models []string
	if b := h.config.EnabledBackend(name); b != nil {
		ctx, cancel := context.WithTimeout(r.Context(), h.config.Admin.BackendModelsTimeout)
		defer cancel()
		var err error
		if models, err = backend.FetchModelIDs(ctx, b.BaseURL, b.APIKey, nil); err != nil {
			h.logger.Warn().Err(err).Str("backend", name).Msg("fetching models")
			models = nil
		}
	}
	writeHTML(w, r, templates.ModelOptions(models, selected))
}

func checkBackendHealth(baseURL string, timeout time.Duration) bool {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (h *Handler) renderBackendList(w http.ResponseWriter, r *http.Request, toast string) {
	writeHTML(w, r, templates.BackendsContent(h.config.Backends), templates.Toast(toast, "success"))
}
