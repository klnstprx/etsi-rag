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

	// Resolve backends and create LLM clients per stage (graceful: nil on failure)
	embeddingClient, err := clientForStage(cfg, "embedding", cfg.Embedding.Backend, llm.ClientConfig{
		Model: cfg.Embedding.Model,
	}, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}

	chatClient, err := clientForStage(cfg, "chat", cfg.Chat.Backend, llm.ClientConfig{
		Model:       cfg.Chat.Model,
		MaxTokens:   int64(cfg.Chat.MaxTokens),
		Temperature: cfg.Chat.Temperature,
	}, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("chat backend not available")
	}

	// Initialize token counter, probe llama.cpp for auto-detection, then create chunker
	embeddingCounter, err := counterForBackend(cfg, cfg.Embedding.Backend, cfg.Embedding.Model, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	probeLlamaCpp(cfg, logger)
	chunker := chunking.NewStructureAwareChunker(cfg.Chunking, embeddingCounter, cfg.Embedding.MaxTokens)

	// Initialize RAG pipeline (may have nil embedder/chatter — guarded in pipeline methods)
	pipeline := rag.NewPipeline(sqliteStore, embeddingClient, chatClient, chunker, embeddingCounter, cfg.RAG)

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

	// Create API handler and router
	handler := api.NewHandler(api.HandlerConfig{
		EmbeddingClient: embeddingClient,
		ChatClient:      chatClient,
		Counter:         embeddingCounter,
		Pipeline:        pipeline,
		Store:           sqliteStore,
		SessionStore:    sessionStore,
	})
	router := api.NewRouter(handler, logger)

	// Create agent (only if [agent] backend is configured)
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		agentClient, err := clientForStage(cfg, "agent", cfg.Agent.Backend, llm.ClientConfig{
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
			agentInstance = agent.NewAgent(agentClient, tools, prompts.Agent, logger)
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent initialized")
		}
	} else {
		logger.Warn().Msg("no [agent] backend configured, agent disabled")
	}

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
		rebuildClients(cfg, pipeline, handler, adminHandler, sqliteStore, logger)
	})

	// Mount admin routes
	router.Mount("/admin", adminHandler.Routes())

	// Create server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 120 * time.Second, // Long timeout for LLM responses
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

// clientForStage resolves a backend by name from the config and builds an LLM client.
// The caller provides stage-specific fields (model, max_tokens, temperature) in partialCfg;
// BaseURL and APIKey are filled from the backend.
func clientForStage(cfg *config.Config, stage, backendName string, partialCfg llm.ClientConfig, logger zerolog.Logger) (*llm.Client, error) {
	b := cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for stage %s", backendName, stage)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	logger.Info().Str("stage", stage).Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewClient(partialCfg), nil
}

// rebuildClients rebuilds all LLM clients from the live config and swaps them
// into pipeline, API handler, and admin handler. Called after every config save.
func rebuildClients(
	cfg *config.Config,
	pipeline *rag.Pipeline,
	apiHandler *api.Handler,
	adminHandler *admin.Handler,
	st *store.SQLiteStore,
	logger zerolog.Logger,
) {
	logger.Info().Msg("rebuilding LLM clients after config change")

	// Embedding
	if ec, err := clientForStage(cfg, "embedding", cfg.Embedding.Backend, llm.ClientConfig{
		Model: cfg.Embedding.Model,
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
	probeLlamaCpp(cfg, logger)
	pipeline.SetCounter(counter)
	apiHandler.SetCounter(counter)
	pipeline.SetChunker(chunking.NewStructureAwareChunker(cfg.Chunking, counter, cfg.Embedding.MaxTokens))

	// Chat
	if cc, err := clientForStage(cfg, "chat", cfg.Chat.Backend, llm.ClientConfig{
		Model:       cfg.Chat.Model,
		MaxTokens:   int64(cfg.Chat.MaxTokens),
		Temperature: cfg.Chat.Temperature,
	}, logger); err != nil {
		logger.Warn().Err(err).Msg("chat client rebuild failed")
		pipeline.SetChatter(nil)
		apiHandler.SetChatClient(nil)
	} else {
		pipeline.SetChatter(cc)
		apiHandler.SetChatClient(cc)
	}

	// Agent
	if cfg.Agent.Backend != "" {
		if ac, err := clientForStage(cfg, "agent", cfg.Agent.Backend, llm.ClientConfig{
			Model:       cfg.Agent.Model,
			MaxTokens:   int64(cfg.Agent.MaxTokens),
			Temperature: cfg.Agent.Temperature,
		}, logger); err != nil {
			logger.Warn().Err(err).Msg("agent client rebuild failed")
			adminHandler.SetAgent(nil)
		} else {
			tools := []agent.Tool{
				agent.NewRAGSearchTool(pipeline),
				agent.NewListDocumentsTool(st),
			}
			adminHandler.SetAgent(agent.NewAgent(ac, tools, prompts.Agent, logger))
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent rebuilt")
		}
	} else {
		adminHandler.SetAgent(nil)
	}
}

// probeLlamaCpp queries a llama.cpp backend's /props and /v1/models endpoints
// to auto-detect embedding context window and dimension, updating cfg in-place.
func probeLlamaCpp(cfg *config.Config, logger zerolog.Logger) {
	b := cfg.GetBackend(cfg.Embedding.Backend)
	if b == nil || !b.IsLlamaCpp {
		return
	}

	props, err := backend.FetchLlamaCppProps(context.Background(), b.BaseURL, nil)
	if err != nil {
		logger.Warn().Err(err).Str("baseURL", b.BaseURL).Msg("could not probe llama.cpp, using config values")
		return
	}

	if props.ContextSize > 0 {
		logger.Info().Int("n_ctx", props.ContextSize).Msg("auto-detected embedding context window")
		cfg.Embedding.MaxTokens = props.ContextSize
	}
	if props.EmbeddingDim > 0 {
		logger.Info().Int("n_embd", props.EmbeddingDim).Msg("auto-detected embedding dimension")
		cfg.Embedding.Dimension = props.EmbeddingDim
	}
	if props.ModelPath != "" {
		logger.Info().Str("model_path", props.ModelPath).Msg("llama.cpp model")
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
		return tokencount.NewLlamaCppCounter(base, nil), nil
	}
	tc, err := tokencount.NewTiktokenCounter(model)
	if err != nil {
		return nil, fmt.Errorf("tiktoken init for model %q: %w", model, err)
	}
	logger.Info().Str("backend", b.Name).Str("model", model).Msg("using tiktoken counter")
	return tc, nil
}
