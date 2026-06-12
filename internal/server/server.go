// Package server wires together all application components and manages the
// HTTP server lifecycle.
package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"etsi-rag/internal/admin"
	"etsi-rag/internal/agent"
	"etsi-rag/internal/api"
	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"

	"github.com/rs/zerolog"
)

// Server holds all live application components.
//
// The embedder and agent are owned by shared handles (single source of truth
// for hot-swapped runtime resources). Retriever and Ingester both reference
// the same EmbedderHandle so the "one embedding model per corpus" invariant
// is enforced by construction; admin and api both reference the same
// AgentHandle so a config-reload swap propagates without parallel SetAgent
// fanout.
type Server struct {
	cfg        *config.Config
	configPath string
	logger     zerolog.Logger

	store          *store.SQLiteStore
	sessionStore   *session.Store
	embedderHandle *llm.EmbedderHandle
	agentHandle    *agent.Handle
	retriever      *rag.Retriever
	ingester       *ingest.Ingester
	sources        *sources.Registry
	crawler        *crawler.Crawler
	diag           *diag.Registry
	probed         *probedProps

	apiHandler   *api.Handler
	adminHandler *admin.Handler
	httpServer   *http.Server

	baseCtx    context.Context
	baseCancel context.CancelFunc
}

// New initializes all application components and returns a ready-to-run Server.
func New(cfg *config.Config, configPath string, logger zerolog.Logger) (*Server, error) {
	logger.Info().
		Str("host", cfg.Server.Host).
		Int("port", cfg.Server.Port).
		Str("db_path", cfg.Database.Path).
		Msg("configuration loaded")

	// Initialize database
	dbPath := cfg.Database.Path
	if dbPath != ":memory:" {
		dir := filepath.Dir(dbPath)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating database directory: %w", err)
		}
	}

	db, err := store.OpenDB(dbPath)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	if err := store.Migrate(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("running migrations: %w", err)
	}

	// probedProps holds auto-detected model/server capacities (probe wins,
	// config is fallback). Constructed BEFORE the store so the dimension
	// closure reads the resolved value (probed → fallback cfg) rather than the
	// raw config field.
	probed := newProbedProps(cfg, logger)
	sqliteStore := store.NewSQLiteStore(db, probed.EmbeddingDimension)

	baseCtx, baseCancel := context.WithCancel(context.Background())
	// IMPORTANT: any non-fatal startup error in this function should also call
	// s.diag.Set so the admin dashboard surfaces it. Logging alone is no longer
	// enough — operators don't tail logs to find a misconfigured backend.
	s := &Server{
		cfg:        cfg,
		configPath: configPath,
		logger:     logger,
		store:      sqliteStore,
		diag:       diag.New(),
		probed:     probed,
		baseCtx:    baseCtx,
		baseCancel: baseCancel,
	}

	s.warmEmbeddingCache(context.Background())

	// Resolve embedding backend and create client (graceful: nil on failure).
	// Re-runnable via rebuildClients on admin save.
	embedName, embedCfg := s.embedClientConfig()
	embeddingClient, err := s.resolveEmbedClient(embedName, embedCfg)
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}
	s.publishProbeResult(issueEmbeddingBackend, err)
	s.embedderHandle = llm.NewEmbedderHandle(embeddingClient)

	// Initialize token counter, probe llama.cpp for auto-detection, then create chunker
	embeddingCounter, err := s.resolveCounter(cfg.Embedding.Backend, cfg.Embedding.Model)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	s.publishProbeResult(issueEmbeddingCounter, err)
	s.probeLlamaCppBackends()
	chunker := chunking.NewStructureAwareChunker(cfg.Chunking, embeddingCounter, s.probed.EmbeddingMaxTokens())

	// Build the retrieval and ingest sides. Both share the same embedder
	// handle so a config-reload swap reaches them in lockstep.
	s.retriever = rag.NewRetriever(sqliteStore, s.embedderHandle, embeddingCounter, cfg.Retrieval)
	s.ingester = ingest.NewIngester(sqliteStore, s.embedderHandle, chunker)

	// Wire optional reranker. Disabled = no-op; resolution failures degrade
	// retrieval to the un-reranked path without crashing the server.
	s.attachReranker()

	objStore := s.attachObjectStore()

	// Sources: in-memory registry seeded from sources.toml, reconciled
	// against any orphaned crawl_pages rows whose source_key disappeared.
	configDir := filepath.Dir(configPath)
	s.loadAndReconcileSources(configDir)

	// Initialize crawler only if dependencies are available.
	if embeddingClient != nil && objStore != nil {
		s.crawler = crawler.New(sqliteStore, objStore, s.ingester, cfg.Crawler, s.sources, logger)
		s.diag.Clear("crawler.disabled")
	} else {
		logger.Warn().Msg("crawler disabled (requires embedding backend + object store)")
		s.warnIssue("crawler.disabled", "crawler", diag.SeverityWarn,
			"Crawler disabled (requires embedding backend + object store)",
			"",
			"Fix the embedding backend and object store, then restart.",
			true)
	}

	// Initialize session store
	s.sessionStore = session.NewStore(cfg.Session)

	// Create agent (only if [agent] backend is configured) and wrap it in
	// the shared handle. admin and api both read from this handle so a
	// rebuild swap is observed by both without parallel SetAgent calls.
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		s.diag.Clear(issueAgentNoBackend.ID)
		chatName, chatCfg := s.chatClientConfig()
		agentClient, err := s.resolveChatClient(chatName, chatCfg)
		if err != nil {
			logger.Warn().Err(err).Msg("agent backend not available, agent disabled")
			s.clearBudgetIssues()
		} else {
			agentInstance = s.buildAgent(agentClient)
			logger.Info().Str("backend", cfg.Agent.Backend).
				Bool("vision_enabled", cfg.Agent.Vision.Enabled).
				Str("public_base_url", cfg.Agent.Vision.PublicBaseURL).
				Msg("agent initialized")
		}
		s.publishProbeResult(issueAgentBackend, err)
	} else {
		logger.Warn().Msg("no [agent] backend configured, agent disabled")
		s.diag.Set(issueAgentNoBackend)
		s.clearBudgetIssues()
	}
	s.agentHandle = agent.NewHandle(agentInstance)

	// Create API handler and router
	s.apiHandler = api.NewHandler(api.HandlerConfig{
		AgentHandle:  s.agentHandle,
		Retriever:    s.retriever,
		Ingester:     s.ingester,
		Store:        sqliteStore,
		SessionStore: s.sessionStore,
		ObjStore:     objStore,
	})
	router := api.NewRouter(s.apiHandler, logger, time.Duration(cfg.Server.RequestTimeout)*time.Second)

	// Create admin handler
	s.adminHandler = admin.NewHandler(admin.HandlerConfig{
		Store:        sqliteStore,
		Config:       cfg,
		ConfigPath:   configPath,
		ConfigDir:    configDir,
		Sources:      s.sources,
		Diag:         s.diag,
		SessionStore: s.sessionStore,
		AgentHandle:  s.agentHandle,
		Crawler:      s.crawler,
		Ingester:     s.ingester,
		ObjStore:     objStore,
		Logger:       logger,
		BaseCtx:      s.baseCtx,
		Capacities:   probed,
	})
	s.adminHandler.SetOnConfigSave(s.rebuildClients)

	// Register diag rechecks so the dashboard "Re-check" button has work to do
	// for the issues whose underlying probes can be re-run in-process.
	s.registerDiagRechecks()

	// Mount admin routes
	router.Mount("/admin", s.adminHandler.Routes())

	// Build HTTP server (not started yet)
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	return s, nil
}

// warmEmbeddingCache loads the chunk embeddings BLOB cache up-front so the
// first query doesn't pay the ~300 ms scan latency. Non-fatal: SearchSimilar
// transparently falls back to a SQL scan when the cache is unloaded. The warm
// path is one-shot at boot — RequiresRestart=true on the dashboard.
func (s *Server) warmEmbeddingCache(ctx context.Context) {
	warmStart := time.Now()
	n, err := s.store.Warm(ctx)
	if err != nil {
		s.logger.Warn().Err(err).Msg("embedding cache warm failed; SearchSimilar will use SQL fallback")
		s.warnIssue("embedding.warm", "embedding", diag.SeverityWarn,
			"Embedding cache failed to warm; falling back to SQL scan",
			err.Error(),
			"Restart the server to retry the cache warm.",
			true)
		return
	}
	s.logger.Info().
		Int("chunks", n).
		Dur("duration", time.Since(warmStart)).
		Msg("embedding cache warmed")
}

// attachObjectStore builds the optional object store client. Failures degrade
// the crawler and image-artifact paths without crashing the server. Returns
// nil on failure; callers must nil-check before passing downstream.
func (s *Server) attachObjectStore() *objectstore.Client {
	objStore, err := objectstore.New(s.cfg.ObjectStore)
	if err != nil {
		s.logger.Warn().Err(err).Msg("object store not available")
		s.warnIssue("objectstore.unavailable", "objectstore", diag.SeverityError,
			"Object store (RustFS) not available; crawler and image artifacts disabled",
			err.Error(),
			"Start RustFS, then restart the server.",
			true)
		return nil
	}
	return objStore
}

// loadAndReconcileSources seeds sources.toml from the default file if absent,
// loads it into the in-memory Registry, and reconciles crawl_pages rows whose
// source_key disappeared since the last load (marks them stale so Purge Stale
// can clear them, matching the admin-UI delete semantics).
func (s *Server) loadAndReconcileSources(configDir string) {
	if err := config.EnsureSourcesFile(configDir); err != nil {
		s.logger.Warn().Err(err).Msg("ensuring sources.toml")
		s.warnIssue("sources.ensure", "sources", diag.SeverityError,
			"Could not seed sources.toml from sources.default.toml",
			err.Error(),
			"Verify the config directory is writable and the default file exists.",
			false)
	} else {
		s.diag.Clear("sources.ensure")
	}

	s.sources = sources.New(config.SourcesPath(configDir), s.logger)
	s.sources.SetDiag(s.diag) // sources.parse / sources.empty are published from here on
	loadErr := s.sources.Load()
	if loadErr != nil {
		s.logger.Warn().Err(loadErr).Msg("loading sources.toml; crawler will start with no sources")
		return
	}

	// Reconcile crawl_pages with the loaded sources: any page whose source_key
	// no longer appears in sources.toml gets marked stale. Skip on parse
	// failure — we don't want a broken file to mass-stale the corpus.
	n, recErr := s.store.MarkOrphanPagesStale(context.Background(), s.sources.KnownKeys())
	if recErr != nil {
		s.logger.Warn().Err(recErr).Msg("reconciling orphan crawl pages")
	} else if n > 0 {
		s.logger.Info().Int("pages", n).Msg("marked orphan crawl pages stale (source removed from sources.toml)")
	}
	s.publishProbeResult(issueSourcesReconcile, recErr)
}

// warnIssue builds and publishes a diag.Issue from inline parameters. Used
// for the one-shot RequiresRestart issues that don't share a template with
// the rebuild-replayable issues in diag.go (those go through publishProbeResult).
func (s *Server) warnIssue(id, component string, severity diag.Severity, message, detail, hint string, requiresRestart bool) {
	s.diag.Set(diag.Issue{
		ID:              id,
		Component:       component,
		Severity:        severity,
		Message:         message,
		Detail:          detail,
		Hint:            hint,
		RequiresRestart: requiresRestart,
	})
}

// Run starts the HTTP server and blocks until a shutdown signal is received.
func (s *Server) Run() error {
	errChan := make(chan error, 1)
	go func() {
		s.logger.Info().Str("addr", s.httpServer.Addr).Msg("starting server")
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		s.logger.Info().Str("signal", sig.String()).Msg("shutting down server")
	}

	s.Close()
	return nil
}

// Close performs graceful shutdown of all components.
func (s *Server) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := s.httpServer.Shutdown(ctx); err != nil {
		s.logger.Warn().Err(err).Msg("server shutdown error")
	}
	// Cancel server-lifetime context so in-flight background goroutines
	// (e.g. crawl runs spawned by admin handlers) observe shutdown.
	s.baseCancel()
	if s.crawler != nil {
		s.crawler.Close()
	}
	s.sessionStore.Close()
	if err := s.store.Close(); err != nil {
		s.logger.Warn().Err(err).Msg("error closing database")
	}

	s.logger.Info().Msg("server stopped")
}
