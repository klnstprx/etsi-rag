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
	cfg.Retrieval.TopK = 5

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Change non-LLM config
	cfg.Retrieval.TopK = 20

	// Should not panic and should propagate changes
	srv.rebuildClients()
}

// TestRebuildClients_PreservesVisionWiring locks the rebuild path against a
// regression where the agent was reconstructed without its VisionGetter or
// without the search tool's Store/MaxImages — silently disabling vision and
// image attachment after any admin save.
func TestRebuildClients_PreservesVisionWiring(t *testing.T) {
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
	cfg.Agent.Vision.Enabled = true
	cfg.Agent.Vision.MaxImagesPerCall = 4
	cfg.Agent.Vision.PublicBaseURL = "http://localhost:3000"

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	// Initial state: vision wired through to the configured backend + URL.
	a := srv.agentHandle.Get()
	if a == nil {
		t.Fatal("agent should be wired after New()")
	}
	if enabled, baseURL := srv.agentVision(); !enabled || baseURL != "http://localhost:3000" {
		t.Fatalf("after New: agentVision()=(%v,%q), want (true,%q)", enabled, baseURL, "http://localhost:3000")
	}

	// Simulate any admin save (e.g. tweaking another config section). The
	// rebuilt agent must keep the same vision wiring.
	srv.rebuildClients()

	a = srv.agentHandle.Get()
	if a == nil {
		t.Fatal("agent should still be wired after rebuildClients()")
	}
	if enabled, baseURL := srv.agentVision(); !enabled || baseURL != "http://localhost:3000" {
		t.Errorf("after rebuild: agentVision()=(%v,%q), want (true,%q)", enabled, baseURL, "http://localhost:3000")
	}

	// Toggling agent.vision.enabled off via the live cfg pointer must take
	// effect without a further rebuild — the getter reads cfg per call.
	cfg.Agent.Vision.Enabled = false
	if enabled, _ := srv.agentVision(); enabled {
		t.Error("agentVision() should observe agent.vision.enabled=false through live cfg")
	}
}

// TestRebuildClients_VisionReportsBaseURLEmpty asserts the diagnostic
// distinction between "agent.vision.enabled off" and "public_base_url empty":
// when Enabled=true but PublicBaseURL is empty, the getter must return
// (true, "") so the agent can log the actual misconfiguration instead of
// misattributing it to the enabled toggle.
func TestRebuildClients_VisionReportsBaseURLEmpty(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxTokens = 1024
	cfg.Agent.MaxIterations = 5
	cfg.Agent.Vision.Enabled = true
	// PublicBaseURL intentionally left empty.

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, t.TempDir()+"/config.toml", logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() { srv.store.Close(); srv.sessionStore.Close() })

	a := srv.agentHandle.Get()
	if a == nil {
		t.Fatal("agent should be wired")
	}
	enabled, baseURL := srv.agentVision()
	if !enabled {
		t.Error("agentVision().enabled should be true when agent.vision.enabled=true even with empty PublicBaseURL — distinguishes the two failure modes for diagnostic logging")
	}
	if baseURL != "" {
		t.Errorf("agentVision().baseURL = %q, want empty", baseURL)
	}
}
