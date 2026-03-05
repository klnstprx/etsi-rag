package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoad(t *testing.T) {
	// Get the config directory relative to this test file
	configDir := filepath.Join("..", "..", "config")

	cfg, err := Load(configDir)
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	// Test server settings
	if cfg.Server.Port != 3000 {
		t.Errorf("expected server port 3000, got %d", cfg.Server.Port)
	}
	if cfg.Server.Host != "127.0.0.1" {
		t.Errorf("expected server host 127.0.0.1, got %s", cfg.Server.Host)
	}

	// Test embedding settings
	if cfg.Embedding.Dimension != 4096 {
		t.Errorf("expected embedding dimension 4096, got %d", cfg.Embedding.Dimension)
	}
	if cfg.Embedding.Backend != "local-llama" {
		t.Errorf("unexpected embedding backend: %s", cfg.Embedding.Backend)
	}

	// Test chat settings
	if cfg.Chat.MaxTokens != 1024 {
		t.Errorf("expected chat max_tokens 1024, got %d", cfg.Chat.MaxTokens)
	}

	// Test agent settings
	if cfg.Agent.Backend != "local-llama" {
		t.Errorf("unexpected agent backend: %s", cfg.Agent.Backend)
	}

	// Test RAG settings
	if cfg.RAG.TopK != 5 {
		t.Errorf("expected top_k 5, got %d", cfg.RAG.TopK)
	}

	// Test session settings
	if cfg.Session.MaxMessages != 10 {
		t.Errorf("expected max_messages 10, got %d", cfg.Session.MaxMessages)
	}
	if cfg.Session.TTLSeconds != 3600 {
		t.Errorf("expected ttl_seconds 3600, got %d", cfg.Session.TTLSeconds)
	}
}

func TestLoadWithEnvOverride(t *testing.T) {
	configDir := filepath.Join("..", "..", "config")

	// Set environment variable override
	os.Setenv("APP__SERVER__PORT", "9000")
	defer os.Unsetenv("APP__SERVER__PORT")

	cfg, err := Load(configDir)
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	if cfg.Server.Port != 9000 {
		t.Errorf("expected server port 9000 from env override, got %d", cfg.Server.Port)
	}
}
