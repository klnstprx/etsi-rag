package server

import (
	"os"
	"testing"

	"etsi-rag/internal/config"

	"github.com/rs/zerolog"
)

func TestNew_MinimalConfig(t *testing.T) {
	cfg := defaultConfig()

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	if srv.store == nil {
		t.Error("expected non-nil store")
	}
	if srv.pipeline == nil {
		t.Error("expected non-nil pipeline")
	}
	if srv.sessionStore == nil {
		t.Error("expected non-nil sessionStore")
	}
	if srv.apiHandler == nil {
		t.Error("expected non-nil apiHandler")
	}
	if srv.adminHandler == nil {
		t.Error("expected non-nil adminHandler")
	}
	if srv.httpServer == nil {
		t.Error("expected non-nil httpServer")
	}
	// No backends configured: crawler and agent should be absent
	if srv.crawler != nil {
		t.Error("expected nil crawler with no backends")
	}
}

func TestNew_WithBackend(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test-backend", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test-backend"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "test-backend"
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

	// Pipeline should exist with resolved embedding client
	if srv.pipeline == nil {
		t.Error("expected non-nil pipeline")
	}
	// No object store → no crawler even with embedding
	if srv.crawler != nil {
		t.Error("expected nil crawler without object store")
	}
}

func TestNew_ServerAddr(t *testing.T) {
	cfg := defaultConfig()
	cfg.Server.Host = "127.0.0.1"
	cfg.Server.Port = 9876

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	want := "127.0.0.1:9876"
	if srv.httpServer.Addr != want {
		t.Errorf("httpServer.Addr = %q, want %q", srv.httpServer.Addr, want)
	}
}
