package admin

import (
	"context"
	"embed"
	"io/fs"
	"net/http"
	"sync/atomic"

	"github.com/a-h/templ"
	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"
)

//go:embed static/styles.css static/htmx.min.js static/htmx-sse.js static/htmx-remove-me.js
var staticFiles embed.FS

// Handler serves the admin UI.
type Handler struct {
	store        *store.SQLiteStore
	config       *config.Config
	configPath   string
	configDir    string // directory holding config.toml and sources.toml
	sources      *sources.Registry
	diag         *diag.Registry
	sessionStore *session.Store
	agent        atomic.Pointer[agent.Agent]
	crawler      *crawler.Crawler
	pipeline     *rag.Pipeline
	objStore     *objectstore.Client
	logger       zerolog.Logger
	onConfigSave func()
	baseCtx      context.Context
	chatStreams  *chatStreamManager
}

// HandlerConfig holds dependencies for the admin handler.
type HandlerConfig struct {
	Store        *store.SQLiteStore
	Config       *config.Config
	ConfigPath   string
	ConfigDir    string
	Sources      *sources.Registry
	Diag         *diag.Registry
	SessionStore *session.Store
	Crawler      *crawler.Crawler
	Pipeline     *rag.Pipeline
	ObjStore     *objectstore.Client
	Logger       zerolog.Logger
	OnConfigSave func()
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
		configDir:    cfg.ConfigDir,
		sources:      cfg.Sources,
		diag:         cfg.Diag,
		sessionStore: cfg.SessionStore,
		crawler:      cfg.Crawler,
		pipeline:     cfg.Pipeline,
		objStore:     cfg.ObjStore,
		logger:       cfg.Logger,
		onConfigSave: cfg.OnConfigSave,
		baseCtx:      baseCtx,
		chatStreams:  newChatStreamManager(),
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
	r.Post("/chat/send", h.ChatSend)
	r.Get("/chat/stream/{sessionID}", h.ChatStream)
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
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if isHTMX(r) {
		fragment.Render(r.Context(), w)
	} else {
		page.Render(r.Context(), w)
	}
}
