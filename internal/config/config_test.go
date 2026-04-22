package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeTempConfig creates a temporary config directory with the given TOML as
// both config.default.toml and config.toml, then returns the directory path.
func writeTempConfig(t *testing.T, toml string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{DefaultConfigFileName, ConfigFileName} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(toml), 0o644); err != nil {
			t.Fatalf("writing %s: %v", name, err)
		}
	}
	return dir
}

func TestLoadMinimal(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[server]
host = "0.0.0.0"
port = 8080
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Explicit values
	if cfg.Server.Host != "0.0.0.0" {
		t.Errorf("host = %q, want 0.0.0.0", cfg.Server.Host)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("port = %d, want 8080", cfg.Server.Port)
	}
}

func TestLoad_RejectsChunkSizeLargerThanContextBudget(t *testing.T) {
	// Invariant: no single chunk should be able to starve the retrieval budget.
	// If max_chunk_size > rag.max_context_tokens, BuildContext would either drop
	// the chunk entirely or need lossy truncation — both bad. Fail loudly at load.
	_, err := Load(writeTempConfig(t, `
[chunking]
max_chunk_size = 4096

[rag]
max_context_tokens = 2048
`))
	if err == nil {
		t.Fatal("expected Load to reject max_chunk_size > max_context_tokens")
	}
}

func TestLoadAppliesDefaults(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	tests := []struct {
		name string
		got  int
		want int
	}{
		{"server.read_timeout", cfg.Server.ReadTimeout, 15},
		{"server.write_timeout", cfg.Server.WriteTimeout, 120},
		{"agent.max_tokens", cfg.Agent.MaxTokens, 1024},
		{"agent.max_iterations", cfg.Agent.MaxIterations, 5},
		{"embedding.max_tokens", cfg.Embedding.MaxTokens, 512},
		{"rag.top_k", cfg.RAG.TopK, 10},
		{"rag.max_context_tokens", cfg.RAG.MaxContextTokens, 2048},
		{"chunking.max_chunk_size", cfg.Chunking.MaxChunkSize, 256},
		{"chunking.chunk_overlap", cfg.Chunking.ChunkOverlap, 32},
		{"chunking.min_chunk_size", cfg.Chunking.MinChunkSize, 64},
		{"session.max_messages", cfg.Session.MaxMessages, 10},
		{"session.ttl_seconds", cfg.Session.TTLSeconds, 3600},
		{"crawler.max_depth", cfg.Crawler.MaxDepth, 10},
		{"crawler.max_pages", cfg.Crawler.MaxPages, 5000},
		{"crawler.concurrency", cfg.Crawler.Concurrency, 4},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
		}
	}

	if cfg.Agent.Temperature != 0.7 {
		t.Errorf("agent.temperature = %f, want 0.7", cfg.Agent.Temperature)
	}
	if cfg.ObjectStore.Region != "us-east-1" {
		t.Errorf("object_store.region = %q, want us-east-1", cfg.ObjectStore.Region)
	}
	if len(cfg.Chunking.Separators) != 5 {
		t.Errorf("chunking.separators length = %d, want 5", len(cfg.Chunking.Separators))
	}

	durationTests := []struct {
		name string
		got  time.Duration
		want time.Duration
	}{
		{"server.idle_timeout", cfg.Server.IdleTimeout, 60 * time.Second},
		{"server.shutdown_timeout", cfg.Server.ShutdownTimeout, 30 * time.Second},
		{"admin.backend_probe_timeout", cfg.Admin.BackendProbeTimeout, 3 * time.Second},
		{"admin.backend_models_timeout", cfg.Admin.BackendModelsTimeout, 5 * time.Second},
		{"admin.chat_stream_timeout", cfg.Admin.ChatStreamTimeout, 5 * time.Minute},
		{"crawler.progress_log_interval", cfg.Crawler.ProgressLogInterval, 30 * time.Second},
	}
	for _, tt := range durationTests {
		if tt.got != tt.want {
			t.Errorf("%s = %s, want %s", tt.name, tt.got, tt.want)
		}
	}
}

func TestLoadExplicitOverridesDefaults(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[server]
read_timeout = 30

[agent]
max_tokens = 4096
temperature = 0.3
max_iterations = 10

[embedding]
max_tokens = 2048

[chunking]
max_chunk_size = 512
chunk_overlap = 64
min_chunk_size = 128

[crawler]
max_depth = 20
concurrency = 8
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	tests := []struct {
		name string
		got  int
		want int
	}{
		{"server.read_timeout", cfg.Server.ReadTimeout, 30},
		{"agent.max_tokens", cfg.Agent.MaxTokens, 4096},
		{"agent.max_iterations", cfg.Agent.MaxIterations, 10},
		{"embedding.max_tokens", cfg.Embedding.MaxTokens, 2048},
		{"chunking.max_chunk_size", cfg.Chunking.MaxChunkSize, 512},
		{"chunking.chunk_overlap", cfg.Chunking.ChunkOverlap, 64},
		{"chunking.min_chunk_size", cfg.Chunking.MinChunkSize, 128},
		{"crawler.max_depth", cfg.Crawler.MaxDepth, 20},
		{"crawler.concurrency", cfg.Crawler.Concurrency, 8},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
		}
	}
	if cfg.Agent.Temperature != 0.3 {
		t.Errorf("agent.temperature = %f, want 0.3", cfg.Agent.Temperature)
	}
}

func TestLoadEnvOverride(t *testing.T) {
	t.Setenv("APP__SERVER__PORT", "9000")
	t.Setenv("APP__RAG__TOP_K", "10")

	cfg, err := Load(writeTempConfig(t, `
[server]
port = 3000

[rag]
top_k = 5
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 9000 {
		t.Errorf("port = %d, want 9000 (env override)", cfg.Server.Port)
	}
	if cfg.RAG.TopK != 10 {
		t.Errorf("top_k = %d, want 10 (env override)", cfg.RAG.TopK)
	}
}

func TestLoadModeOverride(t *testing.T) {
	dir := writeTempConfig(t, `
[server]
port = 3000
`)
	// Write a mode-specific override file
	modeToml := `
[server]
port = 3001
`
	if err := os.WriteFile(filepath.Join(dir, "test.toml"), []byte(modeToml), 0o644); err != nil {
		t.Fatalf("writing test.toml: %v", err)
	}

	t.Setenv("RUN_MODE", "test")

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 3001 {
		t.Errorf("port = %d, want 3001 (mode override)", cfg.Server.Port)
	}
}

func TestLoadBackends(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[[backends]]
name = "llama"
base_url = "http://localhost:8080/v1"
api_key = "sk-test"
enabled = true
is_llama_cpp = true

[[backends]]
name = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-prod"
enabled = false
is_llama_cpp = false
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if len(cfg.Backends) != 2 {
		t.Fatalf("backends count = %d, want 2", len(cfg.Backends))
	}

	b := cfg.GetBackend("llama")
	if b == nil {
		t.Fatal("GetBackend(llama) = nil")
	}
	if !b.IsLlamaCpp {
		t.Error("llama backend should be is_llama_cpp=true")
	}

	// Disabled backend should not be returned
	if cfg.GetBackend("openai") != nil {
		t.Error("GetBackend(openai) should be nil (disabled)")
	}

	// Non-existent backend
	if cfg.GetBackend("missing") != nil {
		t.Error("GetBackend(missing) should be nil")
	}
}

func TestLoadCrawlerConfig(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[crawler]
max_body_size = 5242880
js_patterns = ["/subject/", "/grades/"]
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Crawler.MaxBodySize != 5242880 {
		t.Errorf("max_body_size = %d, want 5242880", cfg.Crawler.MaxBodySize)
	}
	if len(cfg.Crawler.JSPatterns) != 2 {
		t.Fatalf("js_patterns length = %d, want 2", len(cfg.Crawler.JSPatterns))
	}
	if cfg.Crawler.JSPatterns[0] != "/subject/" {
		t.Errorf("js_patterns[0] = %q, want /subject/", cfg.Crawler.JSPatterns[0])
	}
}

func TestEnsureConfigFileCopiesDefault(t *testing.T) {
	dir := t.TempDir()

	// Only write config.default.toml — config.toml should be created by Load
	defaultToml := `
[server]
port = 7777
`
	if err := os.WriteFile(filepath.Join(dir, DefaultConfigFileName), []byte(defaultToml), 0o644); err != nil {
		t.Fatalf("writing default config: %v", err)
	}

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 7777 {
		t.Errorf("port = %d, want 7777 (copied from default)", cfg.Server.Port)
	}

	// config.toml should now exist
	if _, err := os.Stat(filepath.Join(dir, ConfigFileName)); err != nil {
		t.Errorf("config.toml should have been created: %v", err)
	}
}

func TestWriteTo(t *testing.T) {
	cfg := &Config{
		Server: ServerConfig{Host: "0.0.0.0", Port: 9090},
		RAG:    RAGConfig{TopK: 3},
	}

	path := filepath.Join(t.TempDir(), "out.toml")
	if err := WriteTo(cfg, path); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}

	// Read it back
	dir := t.TempDir()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading written config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, DefaultConfigFileName), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ConfigFileName), data, 0o644); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(dir)
	if err != nil {
		t.Fatalf("Load round-trip: %v", err)
	}
	if loaded.Server.Host != "0.0.0.0" {
		t.Errorf("host = %q, want 0.0.0.0", loaded.Server.Host)
	}
	if loaded.Server.Port != 9090 {
		t.Errorf("port = %d, want 9090", loaded.Server.Port)
	}
	if loaded.RAG.TopK != 3 {
		t.Errorf("top_k = %d, want 3", loaded.RAG.TopK)
	}
}

func TestMaskAPIKey(t *testing.T) {
	tests := []struct {
		key  string
		want string
	}{
		{"", "(none)"},
		{"abc", "••••"},
		{"sk-1234567890", "••••7890"},
	}
	for _, tt := range tests {
		if got := MaskAPIKey(tt.key); got != tt.want {
			t.Errorf("MaskAPIKey(%q) = %q, want %q", tt.key, got, tt.want)
		}
	}
}
