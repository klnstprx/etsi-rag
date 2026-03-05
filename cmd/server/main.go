package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/rs/zerolog"

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

	logger.Info().
		Str("host", cfg.Server.Host).
		Int("port", cfg.Server.Port).
		Str("db_path", cfg.VectorStore.Path).
		Msg("configuration loaded")

	// Initialize database
	dbPath := cfg.VectorStore.Path
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
		EmbeddingModel: cfg.Embedding.Model,
	}, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("embedding backend not available")
	}

	chatClient, err := clientForStage(cfg, "chat", cfg.Chat.Backend, llm.ClientConfig{
		ChatModel:   cfg.Chat.Model,
		MaxTokens:   int64(cfg.Chat.MaxTokens),
		Temperature: cfg.Chat.Temperature,
	}, logger)
	if err != nil {
		logger.Warn().Err(err).Msg("chat backend not available")
	}

	// Initialize chunker
	chunker := chunking.NewRecursiveCharacterChunker(cfg.Chunking)

	// Initialize RAG pipeline (may have nil embedder/chatter — guarded in pipeline methods)
	pipeline := rag.NewPipeline(sqliteStore, embeddingClient, chatClient, chunker, cfg.RAG)

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
		Pipeline:        pipeline,
		Store:           sqliteStore,
		SessionStore:    sessionStore,
	})
	router := api.NewRouter(handler, logger)

	// Create agent (only if [agent] backend is configured)
	var agentInstance *agent.Agent
	if cfg.Agent.Backend != "" {
		agentClient, err := clientForStage(cfg, "agent", cfg.Agent.Backend, llm.ClientConfig{
			ChatModel:   cfg.Agent.Model,
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
			agentInstance = agent.NewAgent(agentClient, tools, agentSystemPrompt, logger)
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent initialized")
		}
	} else {
		logger.Warn().Msg("no [agent] backend configured, agent disabled")
	}

	// Create admin handler
	configPath := filepath.Join(configDir, "default.toml")
	adminHandler := admin.NewHandler(admin.HandlerConfig{
		Store:        sqliteStore,
		Config:       cfg,
		ConfigPath:   configPath,
		SessionStore: sessionStore,
		Crawler:      crawlerInstance,
		Pipeline:     pipeline,
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
		EmbeddingModel: cfg.Embedding.Model,
	}, logger); err != nil {
		logger.Warn().Err(err).Msg("embedding client rebuild failed")
		pipeline.SetEmbedder(nil)
		apiHandler.SetEmbeddingClient(nil)
	} else {
		pipeline.SetEmbedder(ec)
		apiHandler.SetEmbeddingClient(ec)
	}

	// Chat
	if cc, err := clientForStage(cfg, "chat", cfg.Chat.Backend, llm.ClientConfig{
		ChatModel:   cfg.Chat.Model,
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
			ChatModel:   cfg.Agent.Model,
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
			adminHandler.SetAgent(agent.NewAgent(ac, tools, agentSystemPrompt, logger))
			logger.Info().Str("backend", cfg.Agent.Backend).Msg("agent rebuilt")
		}
	} else {
		adminHandler.SetAgent(nil)
	}
}

const agentSystemPrompt = `You are a helpful assistant for ETSI Informatica at the University of Malaga.
You have access to a RAG knowledge base with information about degrees, courses, procedures, and services.
Use the rag_search tool to find relevant information before answering questions.
Use the list_documents tool to see what documents are available.
Always base your answers on the information found in the knowledge base when possible.
If you cannot find relevant information, say so honestly.
Respond in the same language as the user's question.`
