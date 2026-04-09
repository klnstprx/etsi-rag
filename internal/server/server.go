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
	"etsi-rag/internal/llm"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
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
	crawler      *crawler.Crawler

	apiHandler   *api.Handler
	adminHandler *admin.Handler
	httpServer   *http.Server
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

	if err := store.Migrate(db, cfg.Embedding.Dimension); err != nil {
		db.Close()
		return nil, fmt.Errorf("running migrations: %w", err)
	}

	sqliteStore := store.NewSQLiteStore(db, cfg.Embedding.Dimension)

	s := &Server{
		cfg:        cfg,
		configPath: configPath,
		logger:     logger,
		store:      sqliteStore,
	}

	// Resolve embedding backend and create client (graceful: nil on failure)
	embeddingClient, err := s.resolveEmbedClient(cfg.Embedding.Backend, llm.EmbedClientConfig{
		Model:     cfg.Embedding.Model,
		BatchSize: cfg.Embedding.BatchSize,
	})
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}

	// Initialize token counter, probe llama.cpp for auto-detection, then create chunker
	embeddingCounter, err := s.resolveCounter(cfg.Embedding.Backend, cfg.Embedding.Model)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	s.probeLlamaCppBackends()
	chunker := chunking.NewStructureAwareChunker(cfg.Chunking, embeddingCounter, cfg.Embedding.MaxTokens)

	// Initialize RAG pipeline (may have nil embedder — guarded in pipeline methods)
	s.pipeline = rag.NewPipeline(sqliteStore, embeddingClient, chunker, embeddingCounter, cfg.RAG)

	// Initialize object store (graceful: nil on failure)
	objStore, err := objectstore.New(cfg.ObjectStore)
	if err != nil {
		logger.Warn().Err(err).Msg("object store not available")
		objStore = nil
	}

	// Initialize crawler only if dependencies are available
	if embeddingClient != nil && objStore != nil {
		s.crawler = crawler.New(sqliteStore, objStore, s.pipeline, cfg.Crawler, logger)
	} else {
		logger.Warn().Msg("crawler disabled (requires embedding backend + object store)")
	}

	// Seed default crawl sources (idempotent)
	for _, seed := range crawler.DefaultSeeds() {
		if err := sqliteStore.CreateCrawlSource(context.Background(), &seed); err != nil {
			logger.Warn().Err(err).Str("seed", seed.Name).Msg("failed to seed crawl source")
		}
	}

	// Initialize session store
	s.sessionStore = session.NewStore(cfg.Session)

	// Create agent (only if [agent] backend is configured)
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		agentClient, err := s.resolveChatClient(cfg.Agent.Backend, llm.ChatClientConfig{
			Model:       cfg.Agent.Model,
			MaxTokens:   int64(cfg.Agent.MaxTokens),
			Temperature: cfg.Agent.Temperature,
		})
		if err != nil {
			logger.Warn().Err(err).Msg("agent backend not available, agent disabled")
		} else {
			tools := []agent.Tool{
				agent.NewSemanticSearchTool(s.pipeline),
				agent.NewListDocumentsTool(sqliteStore),
			}
			systemPrompt := s.agentSystemPrompt()
			agentCounter, err := s.resolveCounter(cfg.Agent.Backend, cfg.Agent.Model)
			if err != nil {
				logger.Warn().Err(err).Msg("agent token counter not available, tool output guards disabled")
			}
			agentInstance = agent.NewAgent(agentClient, tools, systemPrompt, cfg.Agent.MaxIterations, agentCounter, cfg.Agent.ContextWindow, s.sessionStore, logger)
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent initialized")
		}
	} else {
		logger.Warn().Msg("no [agent] backend configured, agent disabled")
	}

	// Create API handler and router
	s.apiHandler = api.NewHandler(api.HandlerConfig{
		EmbeddingClient: embeddingClient,
		Counter:         embeddingCounter,
		Agent:           agentInstance,
		Pipeline:        s.pipeline,
		Store:           sqliteStore,
		SessionStore:    s.sessionStore,
		Passthrough:     cfg.API.Passthrough,
	})
	router := api.NewRouter(s.apiHandler, logger)

	// Create admin handler
	s.adminHandler = admin.NewHandler(admin.HandlerConfig{
		Store:        sqliteStore,
		Config:       cfg,
		ConfigPath:   configPath,
		SessionStore: s.sessionStore,
		Crawler:      s.crawler,
		Pipeline:     s.pipeline,
		ObjStore:     objStore,
		Logger:       logger,
	})
	s.adminHandler.SetAgent(agentInstance)
	s.adminHandler.SetOnConfigSave(s.rebuildClients)

	// Mount admin routes
	router.Mount("/admin", s.adminHandler.Routes())

	// Build HTTP server (not started yet)
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  60 * time.Second,
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
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := s.httpServer.Shutdown(ctx); err != nil {
		s.logger.Warn().Err(err).Msg("server shutdown error")
	}
	if s.crawler != nil {
		s.crawler.Close()
	}
	s.sessionStore.Close()
	if err := s.store.Close(); err != nil {
		s.logger.Warn().Err(err).Msg("error closing database")
	}

	s.logger.Info().Msg("server stopped")
}
