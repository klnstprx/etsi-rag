package server

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
)

// startupTestServer builds a server in a tempdir, optionally seeding
// sources.toml. It exists so multiple wiring tests can poke at the diag
// registry without re-stating the New() boilerplate.
func startupTestServer(t *testing.T, cfg *config.Config, sourcesTOML string) *Server {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.toml")
	if sourcesTOML != "" {
		if err := os.WriteFile(filepath.Join(dir, "sources.toml"), []byte(sourcesTOML), 0o644); err != nil {
			t.Fatalf("seeding sources.toml: %v", err)
		}
	}

	logger := zerolog.New(os.Stderr).Level(zerolog.Disabled)
	srv, err := New(cfg, configPath, logger)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(func() {
		if srv.store != nil {
			srv.store.Close()
		}
		if srv.sessionStore != nil {
			srv.sessionStore.Close()
		}
	})
	return srv
}

func TestDiag_NoBackendIssueOnBoot(t *testing.T) {
	cfg := defaultConfig()
	srv := startupTestServer(t, cfg, "")

	if !srv.diag.Has("agent.no_backend") {
		t.Errorf("expected agent.no_backend issue at boot with empty Agent.Backend; got snapshot=%+v", srv.diag.Snapshot())
	}
}

func TestDiag_NoBackendClearedAfterRebuild(t *testing.T) {
	// Boot with no agent backend → issue set. Add a backend, rebuild → cleared.
	cfg := defaultConfig()
	srv := startupTestServer(t, cfg, "")
	if !srv.diag.Has("agent.no_backend") {
		t.Fatal("precondition: agent.no_backend should be set on boot")
	}

	srv.cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	srv.cfg.Agent.Backend = "test"
	srv.cfg.Agent.Model = "test-model"
	srv.cfg.Agent.MaxTokens = 1024
	srv.cfg.Agent.Temperature = 0.7
	srv.cfg.Agent.MaxIterations = 5

	srv.rebuildClients()

	if srv.diag.Has("agent.no_backend") {
		t.Errorf("agent.no_backend should be cleared after configuring an agent backend; got snapshot=%+v", srv.diag.Snapshot())
	}
}

func TestDiag_SourcesEmptyOnBoot(t *testing.T) {
	cfg := defaultConfig()
	// Empty sources.toml — defaults file exists but contents are empty.
	srv := startupTestServer(t, cfg, "# empty\n")

	if !srv.diag.Has("sources.empty") {
		t.Errorf("expected sources.empty when sources.toml has no entries; got snapshot=%+v", srv.diag.Snapshot())
	}
}

func TestDiag_SourcesParseOnInvalidTOML(t *testing.T) {
	cfg := defaultConfig()
	// Duplicate key — validateSources rejects it.
	bad := `
[[sources]]
key = 'a'
name = 'A'
seed_url = 'https://example.com/'

[[sources]]
key = 'a'
name = 'A2'
seed_url = 'https://example.com/2/'
`
	srv := startupTestServer(t, cfg, bad)

	if !srv.diag.Has("sources.parse") {
		t.Errorf("expected sources.parse on duplicate-key sources.toml; got snapshot=%+v", srv.diag.Snapshot())
	}
}

func TestDiag_RecheckCoversRegisteredIDs(t *testing.T) {
	cfg := defaultConfig()
	srv := startupTestServer(t, cfg, "")

	// All recheck IDs we register should be callable. Call Recheck explicitly
	// for the agent.no_backend ID — that probe should return an error (since
	// the cfg has no backend), confirming Recheck wires through to the
	// registered closure.
	got, ok := srv.diag.Get("agent.no_backend")
	if !ok {
		t.Fatal("precondition: agent.no_backend should be set")
	}
	priorAt := got.At
	srv.diag.Recheck(context.Background(), "agent.no_backend")
	updated, _ := srv.diag.Get("agent.no_backend")
	if !updated.At.After(priorAt) {
		t.Errorf("Recheck did not update At for agent.no_backend (probe was not invoked)")
	}
}

func TestDiag_SourcesEmptyClearedAfterReplace(t *testing.T) {
	cfg := defaultConfig()
	srv := startupTestServer(t, cfg, "# empty\n")
	if !srv.diag.Has("sources.empty") {
		t.Fatal("precondition: sources.empty should be set on boot")
	}

	err := srv.sources.Replace([]domain.CrawlSource{
		{Key: "k", Name: "K", SeedURL: "https://example.com/", Enabled: true},
	})
	if err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if srv.diag.Has("sources.empty") {
		got, _ := srv.diag.Get("sources.empty")
		t.Errorf("sources.empty should clear after Replace with non-empty list; got %+v", got)
	}
	if len(srv.sources.Snapshot()) != 1 {
		t.Errorf("Snapshot len = %d, want 1", len(srv.sources.Snapshot()))
	}
}

// TestDiag_BudgetPublishedAndCleared exercises the buildAgent → publishBudgetIssues
// path end to end: with an agent backend whose effective context window cannot
// fit a full retrieval result, agent.budget is set; removing the backend (or
// fixing the budget) clears it again.
func TestDiag_BudgetPublishedAndCleared(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxIterations = 5
	// Force the budget too small to hold a full retrieval result: a 2000-token
	// window minus 1024 generation and the default reserves leaves a tiny pool,
	// while max_context_tokens asks for 4000.
	cfg.Agent.ContextWindow = 2000
	cfg.Agent.MaxTokens = 1024
	cfg.Agent.HistoryReserveTokens = 1280
	cfg.Agent.UserReserveTokens = 512
	cfg.Retrieval.MaxContextTokens = 4000

	srv := startupTestServer(t, cfg, "")

	if !srv.diag.Has(issueAgentBudget.ID) {
		t.Fatalf("expected agent.budget at boot when pool < max_context_tokens; snapshot=%+v", srv.diag.Snapshot())
	}

	// Removing the agent backend must clear the stale budget error.
	srv.cfg.Agent.Backend = ""
	srv.rebuildClients()
	if srv.diag.Has(issueAgentBudget.ID) {
		t.Errorf("agent.budget should be cleared after removing the agent backend; snapshot=%+v", srv.diag.Snapshot())
	}
}

// TestDiag_BudgetNotSetWhenHealthy confirms the happy path leaves no budget
// issue: a large window comfortably fits max_context_tokens.
func TestDiag_BudgetNotSetWhenHealthy(t *testing.T) {
	cfg := defaultConfig()
	cfg.Backends = []config.BackendConfig{
		{Name: "test", BaseURL: "http://localhost:9999/v1", Enabled: true},
	}
	cfg.Embedding.Backend = "test"
	cfg.Embedding.Model = "test-model"
	cfg.Agent.Backend = "test"
	cfg.Agent.Model = "test-model"
	cfg.Agent.MaxIterations = 5
	cfg.Agent.ContextWindow = 16384
	cfg.Agent.MaxTokens = 1024
	cfg.Retrieval.MaxContextTokens = 2048
	cfg.Retrieval.TopK = 5
	cfg.Chunking.MaxChunkSize = 256

	srv := startupTestServer(t, cfg, "")

	if srv.diag.Has(issueAgentBudget.ID) {
		t.Errorf("agent.budget should not fire with a healthy window; snapshot=%+v", srv.diag.Snapshot())
	}
	if srv.diag.Has(issueRetrievalTopKClip.ID) {
		t.Errorf("retrieval.topk_clip should not fire when top_k chunks fit; snapshot=%+v", srv.diag.Snapshot())
	}
}

// Touch config so the import remains used even if a future test removes the
// last config.* reference.
var _ = config.Config{}
