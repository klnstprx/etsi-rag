package admin

import (
	"embed"
	"io/fs"
	"net/http"

	"github.com/a-h/templ"
	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
)

//go:embed static/styles.css static/htmx.min.js static/htmx-sse.js
var staticFiles embed.FS

// Handler serves the admin UI.
type Handler struct {
	store        *store.SQLiteStore
	config       *config.Config
	configPath   string
	sessionStore *session.Store
	agent        *agent.Agent
	crawler      *crawler.Crawler
	pipeline     *rag.Pipeline
	logger       zerolog.Logger
	onConfigSave func()
}

// HandlerConfig holds dependencies for the admin handler.
type HandlerConfig struct {
	Store        *store.SQLiteStore
	Config       *config.Config
	ConfigPath   string
	SessionStore *session.Store
	Crawler      *crawler.Crawler
	Pipeline     *rag.Pipeline
	Logger       zerolog.Logger
	OnConfigSave func()
}

// NewHandler creates a new admin handler.
func NewHandler(cfg HandlerConfig) *Handler {
	return &Handler{
		store:        cfg.Store,
		config:       cfg.Config,
		configPath:   cfg.ConfigPath,
		sessionStore: cfg.SessionStore,
		crawler:      cfg.Crawler,
		pipeline:     cfg.Pipeline,
		logger:       cfg.Logger,
		onConfigSave: cfg.OnConfigSave,
	}
}

// SetOnConfigSave sets the callback invoked after every successful config save.
func (h *Handler) SetOnConfigSave(fn func()) {
	h.onConfigSave = fn
}

// fireConfigSave calls the onConfigSave callback if set.
func (h *Handler) fireConfigSave() {
	if h.onConfigSave != nil {
		h.onConfigSave()
	}
}

// Routes returns a chi router with all admin routes mounted.
func (h *Handler) Routes() chi.Router {
	r := chi.NewRouter()

	// Static files
	staticFS, _ := fs.Sub(staticFiles, "static")
	r.Handle("/static/*", http.StripPrefix("/admin/static/", http.FileServer(http.FS(staticFS))))

	// Pages
	r.Get("/", h.Dashboard)
	r.Get("/config", h.ConfigPage)
	r.Post("/config/{section}", h.ConfigSave)

	r.Get("/backends", h.BackendList)
	r.Post("/backends", h.BackendCreate)
	r.Get("/backends/{name}", h.BackendGetRow)
	r.Get("/backends/{name}/edit", h.BackendEditForm)
	r.Put("/backends/{name}", h.BackendUpdate)
	r.Delete("/backends/{name}", h.BackendDelete)
	r.Get("/backends/{name}/health", h.BackendHealth)

	r.Get("/crawler", h.CrawlerPage)
	r.Post("/crawler/run", h.CrawlerRun)
	r.Delete("/crawler/run", h.CrawlerStop)
	r.Get("/crawler/status", h.CrawlerStatus)
	r.Post("/crawler/sources", h.SourceCreate)
	r.Put("/crawler/sources/{id}", h.SourceUpdate)
	r.Delete("/crawler/sources/{id}", h.SourceDelete)
	r.Post("/crawler/purge", h.PurgeStale)

	r.Get("/chat", h.ChatPage)
	r.Post("/chat/send", h.ChatSend)
	r.Get("/chat/stream/{sessionID}", h.ChatStream)
	r.Post("/chat/reset", h.ChatReset)

	return r
}

// isHTMX checks if the request is from HTMX.
func isHTMX(r *http.Request) bool {
	return r.Header.Get("HX-Request") != ""
}

// render renders a full page or HTMX fragment.
func render(w http.ResponseWriter, r *http.Request, page, fragment templ.Component) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if isHTMX(r) {
		fragment.Render(r.Context(), w)
	} else {
		page.Render(r.Context(), w)
	}
}
