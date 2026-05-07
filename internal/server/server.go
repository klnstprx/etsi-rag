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
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"

	"github.com/rs/zerolog"
)

// Server holds all live application components.
type Server struct {
	cfg        *config.Config
	configPath string
	logger     zerolog.Logger

	store        *store.SQLiteStore
	sessionStore *session.Store
	pipeline     *rag.Pipeline
	sources      *sources.Registry
	crawler      *crawler.Crawler
	diag         *diag.Registry

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

	sqliteStore := store.NewSQLiteStore(db, func() int { return cfg.Embedding.Dimension })

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
		baseCtx:    baseCtx,
		baseCancel: baseCancel,
	}

	// Warm the in-memory embedding cache so the first user query doesn't pay
	// the ~300 ms full-table BLOB load. Failures are non-fatal: SearchSimilar
	// transparently falls back to a SQL scan when the cache is unloaded. The
	// warm path is one-shot at boot — RequiresRestart=true on the dashboard.
	{
		warmStart := time.Now()
		if n, err := sqliteStore.Warm(context.Background()); err != nil {
			logger.Warn().Err(err).Msg("embedding cache warm failed; SearchSimilar will use SQL fallback")
			s.diag.Set(diag.Issue{
				ID:              "embedding.warm",
				Component:       "embedding",
				Severity:        diag.SeverityWarn,
				Message:         "Embedding cache failed to warm; falling back to SQL scan",
				Detail:          err.Error(),
				Hint:            "Restart the server to retry the cache warm.",
				RequiresRestart: true,
			})
		} else {
			logger.Info().
				Int("chunks", n).
				Dur("duration", time.Since(warmStart)).
				Msg("embedding cache warmed")
		}
	}

	// Resolve embedding backend and create client (graceful: nil on failure).
	// Re-runnable via rebuildClients on admin save.
	embedName, embedCfg := s.embedClientConfig()
	embeddingClient, err := s.resolveEmbedClient(embedName, embedCfg)
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}
	s.publishProbeResult(issueEmbeddingBackend, err)

	// Initialize token counter, probe llama.cpp for auto-detection, then create chunker
	embeddingCounter, err := s.resolveCounter(cfg.Embedding.Backend, cfg.Embedding.Model)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	s.publishProbeResult(issueEmbeddingCounter, err)
	s.probeLlamaCppBackends()
	chunker := chunking.NewStructureAwareChunker(cfg.Chunking, embeddingCounter, cfg.Embedding.MaxTokens)

	// Initialize RAG pipeline (may have nil embedder — guarded in pipeline methods)
	s.pipeline = rag.NewPipeline(sqliteStore, embeddingClient, chunker, embeddingCounter, cfg.Retrieval)

	// Wire optional reranker. Disabled = no-op; resolution failures degrade
	// retrieval to the un-reranked path without crashing the server.
	s.attachReranker()

	// Initialize object store (graceful: nil on failure). Built once at boot;
	// no in-process re-init path today, hence RequiresRestart on the issue.
	objStore, err := objectstore.New(cfg.ObjectStore)
	if err != nil {
		logger.Warn().Err(err).Msg("object store not available")
		objStore = nil
		s.diag.Set(diag.Issue{
			ID:              "objectstore.unavailable",
			Component:       "objectstore",
			Severity:        diag.SeverityError,
			Message:         "Object store (RustFS) not available; crawler and image artifacts disabled",
			Detail:          err.Error(),
			Hint:            "Start RustFS, then restart the server.",
			RequiresRestart: true,
		})
	}

	// Load crawl sources from config/sources.toml. The Registry is the single
	// in-memory source of truth at runtime; admin CRUD calls Replace, the
	// crawler reads Snapshot on each run, the agent's list_sources reads
	// Snapshot on each turn. The first boot copies sources.default.toml.
	configDir := filepath.Dir(configPath)
	if err := config.EnsureSourcesFile(configDir); err != nil {
		logger.Warn().Err(err).Msg("ensuring sources.toml")
		s.diag.Set(diag.Issue{
			ID: "sources.ensure", Component: "sources", Severity: diag.SeverityError,
			Message: "Could not seed sources.toml from sources.default.toml",
			Detail:  err.Error(),
			Hint:    "Verify the config directory is writable and the default file exists.",
		})
	} else {
		s.diag.Clear("sources.ensure")
	}
	s.sources = sources.New(config.SourcesPath(configDir), logger)
	s.sources.SetDiag(s.diag) // sources.parse / sources.empty are published from here on
	loadErr := s.sources.Load()
	if loadErr != nil {
		logger.Warn().Err(loadErr).Msg("loading sources.toml; crawler will start with no sources")
	}

	// Reconcile crawl_pages with the loaded sources: any page whose source_key
	// no longer appears in sources.toml gets marked stale, so a hand-edit that
	// removes a source has the same effect as deleting it via the admin UI.
	// The operator clears the rows by running Purge Stale. Skip on parse
	// failure — we don't want a broken file to mass-stale the corpus.
	if loadErr == nil {
		n, recErr := sqliteStore.MarkOrphanPagesStale(context.Background(), s.sources.KnownKeys())
		if recErr != nil {
			logger.Warn().Err(recErr).Msg("reconciling orphan crawl pages")
		} else if n > 0 {
			logger.Info().Int("pages", n).Msg("marked orphan crawl pages stale (source removed from sources.toml)")
		}
		s.publishProbeResult(issueSourcesReconcile, recErr)
	}

	// Initialize crawler only if dependencies are available
	if embeddingClient != nil && objStore != nil {
		s.crawler = crawler.New(sqliteStore, objStore, s.pipeline, cfg.Crawler, s.sources, logger)
		s.diag.Clear("crawler.disabled")
	} else {
		logger.Warn().Msg("crawler disabled (requires embedding backend + object store)")
		s.diag.Set(diag.Issue{
			ID:              "crawler.disabled",
			Component:       "crawler",
			Severity:        diag.SeverityWarn,
			Message:         "Crawler disabled (requires embedding backend + object store)",
			Hint:            "Fix the embedding backend and object store, then restart.",
			RequiresRestart: true,
		})
	}

	// Initialize session store
	s.sessionStore = session.NewStore(cfg.Session)

	// Create agent (only if [agent] backend is configured)
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		s.diag.Clear(issueAgentNoBackend.ID)
		chatName, chatCfg := s.chatClientConfig()
		agentClient, err := s.resolveChatClient(chatName, chatCfg)
		if err != nil {
			logger.Warn().Err(err).Msg("agent backend not available, agent disabled")
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
	}

	// Create API handler and router
	s.apiHandler = api.NewHandler(api.HandlerConfig{
		EmbeddingClient: embeddingClient,
		Counter:         embeddingCounter,
		Agent:           agentInstance,
		Pipeline:        s.pipeline,
		Store:           sqliteStore,
		SessionStore:    s.sessionStore,
		ObjStore:        objStore,
		Passthrough:     cfg.Agent.Passthrough,
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
		Crawler:      s.crawler,
		Pipeline:     s.pipeline,
		ObjStore:     objStore,
		Logger:       logger,
		BaseCtx:      s.baseCtx,
	})
	s.adminHandler.SetAgent(agentInstance)
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
