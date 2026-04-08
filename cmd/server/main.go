package main

import (
	"context"
	"etsi-rag/internal/admin"
	"etsi-rag/internal/agent"
	"etsi-rag/internal/api"
	"etsi-rag/internal/backend"
	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog"
)

func main() {
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}).
		With().
		Timestamp().
		Logger()

	if err := run(logger); err != nil {
		logger.Fatal().Err(err).Msg("server error")
	}
}

func run(logger zerolog.Logger) error {
	// Load configuration
	configDir := "config"
	cfg, err := config.Load(configDir)
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}
	configPath := config.ConfigPath(configDir)

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
			return fmt.Errorf("creating database directory: %w", err)
		}
	}

	db, err := store.OpenDB(dbPath)
	if err != nil {
		return fmt.Errorf("opening database: %w", err)
	}

	if err := store.Migrate(db, cfg.Embedding.Dimension); err != nil {
		db.Close()
		return fmt.Errorf("running migrations: %w", err)
	}

	sqliteStore := store.NewSQLiteStore(db, cfg.Embedding.Dimension)

	// Resolve embedding backend and create client (graceful: nil on failure)
	embeddingClient, err := embedClientForStage(cfg, cfg.Embedding.Backend, llm.EmbedClientConfig{
		Model:     cfg.Embedding.Model,
		BatchSize: cfg.Embedding.BatchSize,
	}, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}

	// Initialize token counter, probe llama.cpp for auto-detection, then create chunker
	embeddingCounter, err := counterForBackend(cfg, cfg.Embedding.Backend, cfg.Embedding.Model, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	probeLlamaCppBackends(cfg, logger)
	chunker := chunking.NewStructureAwareChunker(cfg.Chunking, embeddingCounter, cfg.Embedding.MaxTokens)

	// Initialize RAG pipeline (may have nil embedder — guarded in pipeline methods)
	pipeline := rag.NewPipeline(sqliteStore, embeddingClient, chunker, embeddingCounter, cfg.RAG)

	// Initialize object store (graceful: nil on failure)
	objStore, err := objectstore.New(cfg.ObjectStore)
	if err != nil {
		logger.Warn().Err(err).Msg("object store not available")
		objStore = nil
	}

	// Initialize crawler only if dependencies are available
	var crawlerInstance *crawler.Crawler
	if embeddingClient != nil && objStore != nil {
		crawlerInstance = crawler.New(sqliteStore, objStore, pipeline, cfg.Crawler, logger)
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
	sessionStore := session.NewStore(cfg.Session)

	// Create agent (only if [agent] backend is configured)
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		agentClient, err := chatClientForStage(cfg, "agent", cfg.Agent.Backend, llm.ChatClientConfig{
			Model:       cfg.Agent.Model,
			MaxTokens:   int64(cfg.Agent.MaxTokens),
			Temperature: cfg.Agent.Temperature,
		}, logger)
		if err != nil {
			logger.Warn().Err(err).Msg("agent backend not available, agent disabled")
		} else {
			tools := []agent.Tool{
				agent.NewRAGSearchTool(pipeline),
				agent.NewListDocumentsTool(sqliteStore),
			}
			systemPrompt := agentSystemPrompt(cfg)
			agentInstance = agent.NewAgent(agentClient, tools, systemPrompt, cfg.Agent.MaxIterations, sessionStore, logger)
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent initialized")
		}
	} else {
		logger.Warn().Msg("no [agent] backend configured, agent disabled")
	}

	// Create API handler and router
	handler := api.NewHandler(api.HandlerConfig{
		EmbeddingClient: embeddingClient,
		Counter:         embeddingCounter,
		Agent:           agentInstance,
		Pipeline:        pipeline,
		Store:           sqliteStore,
		SessionStore:    sessionStore,
		Passthrough:     cfg.API.Passthrough,
	})
	router := api.NewRouter(handler, logger)

	// Create admin handler
	adminHandler := admin.NewHandler(admin.HandlerConfig{
		Store:        sqliteStore,
		Config:       cfg,
		ConfigPath:   configPath,
		SessionStore: sessionStore,
		Crawler:      crawlerInstance,
		Pipeline:     pipeline,
		ObjStore:     objStore,
		Logger:       logger,
	})
	adminHandler.SetAgent(agentInstance)
	adminHandler.SetOnConfigSave(func() {
		rebuildClients(cfg, pipeline, handler, adminHandler, crawlerInstance, sqliteStore, sessionStore, logger)
	})

	// Mount admin routes
	router.Mount("/admin", adminHandler.Routes())

	// Create server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server
	errChan := make(chan error, 1)
	go func() {
		logger.Info().Str("addr", addr).Msg("starting server")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		logger.Info().Str("signal", sig.String()).Msg("shutting down server")
	}

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		return fmt.Errorf("server shutdown: %w", err)
	}

	// Cleanup
	if crawlerInstance != nil {
		crawlerInstance.Close()
	}
	sessionStore.Close()
	if err := sqliteStore.Close(); err != nil {
		logger.Warn().Err(err).Msg("error closing database")
	}

	logger.Info().Msg("server stopped")
	return nil
}

// agentSystemPrompt returns the system prompt for the agent.
// Uses the config value if set, otherwise falls back to the embedded default.
func agentSystemPrompt(cfg *config.Config) string {
	if cfg.Agent.SystemPrompt != "" {
		return cfg.Agent.SystemPrompt
	}
	return prompts.Agent
}

// embedClientForStage resolves a backend by name and builds an embedding client.
func embedClientForStage(cfg *config.Config, backendName string, partialCfg llm.EmbedClientConfig, logger zerolog.Logger) (*llm.EmbedClient, error) {
	b := cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for embedding", backendName)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	logger.Info().Str("stage", "embedding").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewEmbedClient(partialCfg), nil
}

// chatClientForStage resolves a backend by name and builds a chat client.
func chatClientForStage(cfg *config.Config, stage, backendName string, partialCfg llm.ChatClientConfig, logger zerolog.Logger) (*llm.ChatClient, error) {
	b := cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for stage %s", backendName, stage)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	logger.Info().Str("stage", stage).Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewChatClient(partialCfg), nil
}

// rebuildClients rebuilds all LLM clients from the live config and swaps them
// into pipeline, API handler, admin handler, and crawler. Called after every config save.
func rebuildClients(
	cfg *config.Config,
	pipeline *rag.Pipeline,
	apiHandler *api.Handler,
	adminHandler *admin.Handler,
	crawlerInstance *crawler.Crawler,
	st *store.SQLiteStore,
	sessionStore *session.Store,
	logger zerolog.Logger,
) {
	logger.Info().Msg("rebuilding clients after config change")

	// Propagate non-LLM config changes
	pipeline.SetConfig(cfg.RAG)
	apiHandler.SetPassthrough(cfg.API.Passthrough)
	if crawlerInstance != nil {
		crawlerInstance.SetConfig(cfg.Crawler)
	}

	// Embedding
	if ec, err := embedClientForStage(cfg, cfg.Embedding.Backend, llm.EmbedClientConfig{
		Model:     cfg.Embedding.Model,
		BatchSize: cfg.Embedding.BatchSize,
	}, logger); err != nil {
		logger.Warn().Err(err).Msg("embedding client rebuild failed")
		pipeline.SetEmbedder(nil)
		apiHandler.SetEmbeddingClient(nil)
	} else {
		pipeline.SetEmbedder(ec)
		apiHandler.SetEmbeddingClient(ec)
	}

	// Rebuild token counter and chunker
	counter, err := counterForBackend(cfg, cfg.Embedding.Backend, cfg.Embedding.Model, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter rebuild failed (fix via admin UI)")
	}
	probeLlamaCppBackends(cfg, logger)
	pipeline.SetCounter(counter)
	apiHandler.SetCounter(counter)
	pipeline.SetChunker(chunking.NewStructureAwareChunker(cfg.Chunking, counter, cfg.Embedding.MaxTokens))

	// Agent (single chat client for agent)
	if cfg.Agent.Backend != "" {
		if ac, err := chatClientForStage(cfg, "agent", cfg.Agent.Backend, llm.ChatClientConfig{
			Model:       cfg.Agent.Model,
			MaxTokens:   int64(cfg.Agent.MaxTokens),
			Temperature: cfg.Agent.Temperature,
		}, logger); err != nil {
			logger.Warn().Err(err).Msg("agent client rebuild failed")
			adminHandler.SetAgent(nil)
			apiHandler.SetAgent(nil)
		} else {
			tools := []agent.Tool{
				agent.NewRAGSearchTool(pipeline),
				agent.NewListDocumentsTool(st),
			}
			systemPrompt := agentSystemPrompt(cfg)
			a := agent.NewAgent(ac, tools, systemPrompt, cfg.Agent.MaxIterations, sessionStore, logger)
			adminHandler.SetAgent(a)
			apiHandler.SetAgent(a)
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent rebuilt")
		}
	} else {
		adminHandler.SetAgent(nil)
		apiHandler.SetAgent(nil)
	}
}

// probeLlamaCppBackends probes each unique llama.cpp backend once and applies
// the results to the relevant stage configs.
func probeLlamaCppBackends(cfg *config.Config, logger zerolog.Logger) {
	type stageRef struct {
		name    string
		backend string
	}
	stages := []stageRef{
		{"embedding", cfg.Embedding.Backend},
		{"agent", cfg.Agent.Backend},
	}

	probed := make(map[string]*backend.LlamaCppProps)
	for _, s := range stages {
		if s.backend == "" {
			continue
		}
		if _, done := probed[s.backend]; done {
			continue
		}
		b := cfg.GetBackend(s.backend)
		if b == nil || !b.IsLlamaCpp {
			continue
		}
		props, err := backend.FetchLlamaCppProps(context.Background(), b.BaseURL, nil)
		if err != nil {
			logger.Warn().Err(err).Str("backend", b.Name).
				Msg("could not probe llama.cpp backend, using config values")
			continue
		}
		probed[s.backend] = props
		logger.Info().Str("backend", b.Name).
			Int("n_ctx", props.ContextSize).
			Int("n_embd", props.EmbeddingDim).
			Str("model_path", props.ModelPath).
			Msg("probed llama.cpp backend")
	}

	// Apply embedding-specific results
	if props, ok := probed[cfg.Embedding.Backend]; ok {
		if props.ContextSize > 0 {
			cfg.Embedding.MaxTokens = props.ContextSize
		}
		if props.EmbeddingDim > 0 {
			cfg.Embedding.Dimension = props.EmbeddingDim
		}
	}

	// Apply agent context window
	if props, ok := probed[cfg.Agent.Backend]; ok && props.ContextSize > 0 {
		cfg.Agent.ContextWindow = props.ContextSize
		if cfg.Agent.MaxTokens > props.ContextSize {
			logger.Warn().Int("max_tokens", cfg.Agent.MaxTokens).Int("context_window", props.ContextSize).
				Msg("agent max_tokens exceeds context window")
		}
	}
}

// counterForBackend creates an appropriate token counter for the given backend.
func counterForBackend(cfg *config.Config, backendName, model string, logger zerolog.Logger) (tokencount.Counter, error) {
	b := cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for token counter", backendName)
	}
	if b.IsLlamaCpp {
		base := strings.TrimSuffix(b.BaseURL, "/v1")
		logger.Info().Str("backend", b.Name).Msg("using llama.cpp token counter")
		return tokencount.NewLlamaCppCounter(base, model, nil), nil
	}
	tc, err := tokencount.NewTiktokenCounter(model)
	if err != nil {
		return nil, fmt.Errorf("tiktoken init for model %q: %w", model, err)
	}
	logger.Info().Str("backend", b.Name).Str("model", model).Msg("using tiktoken counter")
	return tc, nil
}
