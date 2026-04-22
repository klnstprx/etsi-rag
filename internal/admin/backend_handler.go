package admin

import (
	"context"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/backend"
	"etsi-rag/internal/config"
)

// BackendGetRow handles GET /admin/backends/{name} — returns the display row (used by Cancel).
func (h *Handler) BackendGetRow(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	b := h.config.GetBackend(name)
	if b == nil {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.BackendRow(*b).Render(r.Context(), w)
}

// BackendEditForm handles GET /admin/backends/{name}/edit — returns an inline edit row.
func (h *Handler) BackendEditForm(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	b := h.config.GetBackend(name)
	if b == nil {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.BackendEditRow(*b).Render(r.Context(), w)
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

	if err := config.WriteTo(h.config, h.configPath); err != nil {
		// Roll back
		h.config.Backends = h.config.Backends[:len(h.config.Backends)-1]
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.fireConfigSave()
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

	idx := -1
	for i := range h.config.Backends {
		if h.config.Backends[i].Name == name {
			idx = i
			break
		}
	}
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

	if err := config.WriteTo(h.config, h.configPath); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.fireConfigSave()
	h.renderBackendList(w, r, "Backend updated")
}

// BackendDelete handles DELETE /admin/backends/{name}.
func (h *Handler) BackendDelete(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")

	idx := -1
	for i := range h.config.Backends {
		if h.config.Backends[i].Name == name {
			idx = i
			break
		}
	}
	if idx < 0 {
		http.Error(w, "backend not found", http.StatusNotFound)
		return
	}

	h.config.Backends = append(h.config.Backends[:idx], h.config.Backends[idx+1:]...)

	if err := config.WriteTo(h.config, h.configPath); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	h.fireConfigSave()
	h.logger.Info().Str("name", name).Msg("backend deleted")
	h.renderBackendList(w, r, "Backend deleted")
}

// BackendHealth handles GET /admin/backends/{name}/health.
func (h *Handler) BackendHealth(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")

	b := h.config.GetBackend(name)
	if b == nil {
		templates.StatusBadge(false).Render(r.Context(), w)
		return
	}

	healthy := checkBackendHealth(b.BaseURL, h.config.Admin.BackendProbeTimeout)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.StatusBadge(healthy).Render(r.Context(), w)
}

// BackendModels handles GET /admin/models?backend=X&current=Y — returns <option> elements.
func (h *Handler) BackendModels(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("backend")
	selected := r.URL.Query().Get("current")

	b := h.config.GetBackend(name)
	if b == nil {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.ModelOptions(nil, selected).Render(r.Context(), w)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), h.config.Admin.BackendModelsTimeout)
	defer cancel()

	models, err := backend.FetchModelIDs(ctx, b.BaseURL, b.APIKey, nil)
	if err != nil {
		h.logger.Warn().Err(err).Str("backend", name).Msg("fetching models")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.ModelOptions(nil, selected).Render(r.Context(), w)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.ModelOptions(models, selected).Render(r.Context(), w)
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
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.BackendsContent(h.config.Backends).Render(r.Context(), w)
	if toast != "" {
		templates.Toast(toast, "success").Render(r.Context(), w)
	}
}
