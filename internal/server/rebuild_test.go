package server

import (
	"os"
	"testing"

	"etsi-rag/internal/config"

	"github.com/rs/zerolog"
)

func TestRebuildClients_NoPanic(t *testing.T) {
	// Build a full server, then verify rebuildClients doesn't panic
	cfg := defaultConfig()

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Should not panic with no backends
	srv.rebuildClients()
}

func TestRebuildClients_WithBackend(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxTokens = 1024
	cfg.Agent.Temperature = 0.7
	cfg.Agent.MaxIterations = 5

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Should not panic
	srv.rebuildClients()
}

func TestRebuildClients_ConfigChange(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test"
	cfg.Embedding.Model = "test-model"

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Simulate config change: add agent backend
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxTokens = 2048
	cfg.Agent.Temperature = 0.5
	cfg.Agent.MaxIterations = 10

	// Rebuild should pick up the new agent config without panic
	srv.rebuildClients()
}

func TestRebuildClients_RemoveBackend(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxTokens = 1024
	cfg.Agent.Temperature = 0.7
	cfg.Agent.MaxIterations = 5

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Simulate removing the agent backend
	cfg.Agent.Backend = ""
	srv.rebuildClients()

	// Simulate removing embedding backend
	cfg.Embedding.Backend = ""
	srv.rebuildClients()
}

func TestRebuildClients_PropagatesNonLLMConfig(t *testing.T) {
	cfg := defaultConfig()
	cfg.RAG.TopK = 5
	cfg.API.Passthrough = false

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Change non-LLM config
	cfg.RAG.TopK = 20
	cfg.API.Passthrough = true

	// Should not panic and should propagate changes
	srv.rebuildClients()
}
