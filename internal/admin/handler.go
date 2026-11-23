package admin

import (
	"context"
	"embed"
	"io/fs"
	"net/http"

	"github.com/a-h/templ"
	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/config"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/session"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"
)

//go:embed static/styles.css static/htmx.min.js static/htmx-remove-me.js static/chat.js
var staticFiles embed.FS

// Capacities reports the live effective values for the capacity fields whose
// config semantics are "0 = auto" (agent context window, embedding/reranker
// max tokens). Implemented by the server's runtime probe holder. The config
// page uses it to display what is actually in effect next to the editable
// operator intent, which itself round-trips untouched (a 0 stays 0 on save).
type Capacities interface {
	AgentContextWindow() int
	EmbeddingMaxTokens() int
	RerankerMaxTokens() int
}

// Handler serves the admin UI.
type Handler struct {
	store        *store.SQLiteStore
	config       *config.Config
	configPath   string
	sources      *sources.Registry
	diag         *diag.Registry
	sessionStore *session.Store
	crawler      *crawler.Crawler
	ingester     *ingest.Ingester
	objStore     *objectstore.Client
	logger       zerolog.Logger
	onConfigSave func()
	baseCtx      context.Context
	capacities   Capacities // may be nil (tests); hints then omit effective values
}

// HandlerConfig holds dependencies for the admin handler.
type HandlerConfig struct {
	Store        *store.SQLiteStore
	Config       *config.Config
	ConfigPath   string
	Sources      *sources.Registry
	Diag         *diag.Registry
	SessionStore *session.Store
	Crawler      *crawler.Crawler
	Ingester     *ingest.Ingester
	ObjStore     *objectstore.Client
	Logger       zerolog.Logger
	// OnConfigSave runs after every successful config or backend save.
	OnConfigSave func()
	Capacities   Capacities
	// BaseCtx is a server-lifetime context used for background goroutines
	// (e.g. crawl runs) that outlive the request they were spawned from.
	// Cancelled when the server shuts down.
	BaseCtx context.Context
}

// NewHandler creates a new admin handler.
func NewHandler(cfg HandlerConfig) *Handler {
	baseCtx := cfg.BaseCtx
	if baseCtx == nil {
		baseCtx = context.Background()
	}
	return &Handler{
		store:        cfg.Store,
		config:       cfg.Config,
		configPath:   cfg.ConfigPath,
		sources:      cfg.Sources,
		diag:         cfg.Diag,
		sessionStore: cfg.SessionStore,
		crawler:      cfg.Crawler,
		ingester:     cfg.Ingester,
		objStore:     cfg.ObjStore,
		logger:       cfg.Logger,
		onConfigSave: cfg.OnConfigSave,
		baseCtx:      baseCtx,
		capacities:   cfg.Capacities,
	}
}

// saveConfig persists the live config and then runs the OnConfigSave hook,
// so every save path rebuilds clients the same way.
func (h *Handler) saveConfig() error {
	if err := config.WriteTo(h.config, h.configPath); err != nil {
		return err
	}
	if h.onConfigSave != nil {
		h.onConfigSave()
	}
	return nil
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
	r.Get("/config/{section}", h.ConfigPage)
	r.Post("/config/{section}", h.ConfigSave)

	r.Get("/backends", h.BackendList)
	r.Post("/backends", h.BackendCreate)
	r.Get("/backends/{name}", h.BackendGetRow)
	r.Get("/backends/{name}/edit", h.BackendEditForm)
	r.Put("/backends/{name}", h.BackendUpdate)
	r.Delete("/backends/{name}", h.BackendDelete)
	r.Get("/backends/{name}/health", h.BackendHealth)
	r.Get("/models", h.BackendModels)

	r.Get("/documents", h.DocumentList)
	r.Get("/documents/{id}", h.DocumentDetail)
	r.Delete("/documents/{id}", h.DocumentDelete)

	r.Get("/health/issues", h.HealthIssues)
	r.Post("/health/recheck", h.HealthRecheck)
	r.Get("/health/objectstore", h.HealthObjectStore)

	r.Get("/crawler", h.CrawlerPage)
	r.Post("/crawler/run", h.CrawlerRun)
	r.Post("/crawler/reprocess", h.CrawlerReprocess)
	r.Delete("/crawler/run", h.CrawlerStop)
	r.Get("/crawler/status", h.CrawlerStatus)
	r.Post("/crawler/sources", h.SourceCreate)
	r.Put("/crawler/sources/{key}", h.SourceUpdate)
	r.Delete("/crawler/sources/{key}", h.SourceDelete)
	r.Post("/crawler/purge", h.PurgeStale)

	r.Get("/chat", h.ChatPage)
	r.Post("/chat/reset", h.ChatReset)

	return r
}

// isHTMX checks if the request is from HTMX (but not a boosted navigation).
func isHTMX(r *http.Request) bool {
	return r.Header.Get("HX-Request") != "" && r.Header.Get("HX-Boosted") == ""
}

// render renders a full page or HTMX fragment.
// Boosted navigations (hx-boost) receive the full page so HTMX can swap the
// entire body, which updates both the sidebar active state and the main content.
func render(w http.ResponseWriter, r *http.Request, page, fragment templ.Component) {
	if isHTMX(r) {
		writeHTML(w, r, fragment)
	} else {
		writeHTML(w, r, page)
	}
}

// writeHTML renders components in order as one HTML response.
func writeHTML(w http.ResponseWriter, r *http.Request, components ...templ.Component) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	for _, c := range components {
		c.Render(r.Context(), w)
	}
}

// writeToast renders a single toast, the reply for actions that fail
// without re-rendering their panel.
func writeToast(w http.ResponseWriter, r *http.Request, msg, kind string) {
	writeHTML(w, r, templates.Toast(msg, kind))
}
