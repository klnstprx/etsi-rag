package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/knadh/koanf/parsers/toml"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

// writeTempConfig creates a temporary config directory with the given TOML as
// both config.default.toml and config.toml, then returns the directory path.
func writeTempConfig(t *testing.T, toml string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{DefaultFileName, FileName} {
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
	// If max_chunk_size > retrieval.max_context_tokens, BuildContext would either
	// drop the chunk entirely or need lossy truncation — both bad. Fail loudly at load.
	_, err := Load(writeTempConfig(t, `
[chunking]
max_chunk_size = 4096

[retrieval]
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
		{"server.write_timeout", cfg.Server.WriteTimeout, 300},
		{"server.request_timeout", cfg.Server.RequestTimeout, 120},
		{"agent.max_tokens", cfg.Agent.MaxTokens, 1024},
		{"agent.max_iterations", cfg.Agent.MaxIterations, 20},
		{"agent.history_reserve_tokens", cfg.Agent.HistoryReserveTokens, 1280},
		{"agent.user_reserve_tokens", cfg.Agent.UserReserveTokens, 512},
		{"embedding.max_tokens", cfg.Embedding.MaxTokens, 0}, // 0 = auto (probe); safe floor applied at consumption
		{"retrieval.top_k", cfg.Retrieval.TopK, 10},
		{"retrieval.max_context_tokens", cfg.Retrieval.MaxContextTokens, 4092},
		{"chunking.max_chunk_size", cfg.Chunking.MaxChunkSize, 256},
		{"chunking.chunk_overlap", cfg.Chunking.ChunkOverlap, 32},
		{"chunking.min_chunk_size", cfg.Chunking.MinChunkSize, 64},
		{"session.max_messages", cfg.Session.MaxMessages, 10},
		{"session.ttl_seconds", cfg.Session.TTLSeconds, 3600},
		{"crawler.max_depth", cfg.Crawler.MaxDepth, 10},
		{"crawler.max_pages", cfg.Crawler.MaxPages, 5000},
		{"crawler.concurrency", cfg.Crawler.Concurrency, 4},
		{"crawler.rate_limit_retries", cfg.Crawler.RateLimitRetries, 4},
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
		{"crawler.rate_limit_backoff", cfg.Crawler.RateLimitBackoff, 30 * time.Second},
		{"crawler.rate_limit_max_wait", cfg.Crawler.RateLimitMaxWait, 5 * time.Minute},
	}
	for _, tt := range durationTests {
		if tt.got != tt.want {
			t.Errorf("%s = %s, want %s", tt.name, tt.got, tt.want)
		}
	}
}

// TestLoadRejectionPatternsCoverCMSErrorPage checks the built-in defaults and
// every shipped config, since a config that lists rejection_patterns replaces
// the defaults.
func TestLoadRejectionPatternsCoverCMSErrorPage(t *testing.T) {
	configs := map[string]string{"built-in defaults": ""}
	for _, path := range []string{"../../config/config.default.toml", "../../deploy/config.docker.toml"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		configs[path] = string(data)
	}
	// The text UMA's CMS error template shows, served with status 200.
	const text = "ha ocurrido un error al intentar obtener los datos y no ha sido posible cargar esta página."
	for name, toml := range configs {
		t.Run(name, func(t *testing.T) {
			cfg, err := Load(writeTempConfig(t, toml))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			for _, pattern := range cfg.Crawler.RejectionPatterns {
				if strings.Contains(text, strings.ToLower(pattern)) {
					return
				}
			}
			t.Errorf("no rejection pattern matches the CMS error text; patterns: %q", cfg.Crawler.RejectionPatterns)
		})
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
history_reserve_tokens = 2000
user_reserve_tokens = 800

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
		{"agent.history_reserve_tokens", cfg.Agent.HistoryReserveTokens, 2000},
		{"agent.user_reserve_tokens", cfg.Agent.UserReserveTokens, 800},
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
	t.Setenv("APP__RETRIEVAL__TOP_K", "10")

	cfg, err := Load(writeTempConfig(t, `
[server]
port = 3000

[retrieval]
top_k = 5
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 9000 {
		t.Errorf("port = %d, want 9000 (env override)", cfg.Server.Port)
	}
	if cfg.Retrieval.TopK != 10 {
		t.Errorf("top_k = %d, want 10 (env override)", cfg.Retrieval.TopK)
	}
}

func TestRetrievalModeDefaultsToDense(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Retrieval.RetrievalMode != "dense" {
		t.Errorf("retrieval_mode = %q, want %q", cfg.Retrieval.RetrievalMode, "dense")
	}
}

func TestRetrievalModeFromTOML(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[retrieval]
retrieval_mode = "hybrid"
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Retrieval.RetrievalMode != "hybrid" {
		t.Errorf("retrieval_mode = %q, want %q", cfg.Retrieval.RetrievalMode, "hybrid")
	}
}

func TestRetrievalModeFromEnv(t *testing.T) {
	t.Setenv("APP__RETRIEVAL__RETRIEVAL_MODE", "hybrid")
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Retrieval.RetrievalMode != "hybrid" {
		t.Errorf("retrieval_mode = %q, want %q (env override)", cfg.Retrieval.RetrievalMode, "hybrid")
	}
}

func TestRetrievalModeRejectsUnknown(t *testing.T) {
	_, err := Load(writeTempConfig(t, `
[retrieval]
retrieval_mode = "wat"
`))
	if err == nil {
		t.Fatal("expected error for unknown retrieval_mode, got nil")
	}
	if !strings.Contains(err.Error(), "wat") {
		t.Errorf("error should mention bad value, got: %v", err)
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

	b := cfg.EnabledBackend("llama")
	if b == nil {
		t.Fatal("EnabledBackend(llama) = nil")
	}
	if !b.IsLlamaCpp {
		t.Error("llama backend should be is_llama_cpp=true")
	}

	// Disabled backend should not be returned
	if cfg.EnabledBackend("openai") != nil {
		t.Error("EnabledBackend(openai) should be nil (disabled)")
	}

	// Non-existent backend
	if cfg.EnabledBackend("missing") != nil {
		t.Error("EnabledBackend(missing) should be nil")
	}
}

func TestLoadCrawlerConfig(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, `
[crawler]
max_body_size = 5242880
js_patterns = ["/subject/", "/grades/"]
rate_limit_retries = 2
rate_limit_backoff = '250ms'
rate_limit_max_wait = '2s'
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Crawler.RateLimitRetries != 2 {
		t.Errorf("rate_limit_retries = %d, want 2", cfg.Crawler.RateLimitRetries)
	}
	if cfg.Crawler.RateLimitBackoff != 250*time.Millisecond {
		t.Errorf("rate_limit_backoff = %s, want 250ms", cfg.Crawler.RateLimitBackoff)
	}
	if cfg.Crawler.RateLimitMaxWait != 2*time.Second {
		t.Errorf("rate_limit_max_wait = %s, want 2s", cfg.Crawler.RateLimitMaxWait)
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

func TestLoad_RejectsInvalidRateLimit(t *testing.T) {
	tests := []struct {
		name    string
		toml    string
		wantKey string
	}{
		{"negative retries", "rate_limit_retries = -1", "rate_limit_retries"},
		{"negative backoff", "rate_limit_backoff = '-1s'", "rate_limit_backoff"},
		{"max wait below backoff", "rate_limit_backoff = '30s'\nrate_limit_max_wait = '10s'", "rate_limit_max_wait"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Load(writeTempConfig(t, "[crawler]\n"+tt.toml+"\n"))
			if err == nil {
				t.Fatal("expected Load to reject the rate limit settings")
			}
			if !strings.Contains(err.Error(), tt.wantKey) {
				t.Errorf("error should name %s, got: %v", tt.wantKey, err)
			}
		})
	}
}

func TestEnsureConfigFileCopiesDefault(t *testing.T) {
	dir := t.TempDir()

	// Only write config.default.toml — config.toml should be created by Load
	defaultToml := `
[server]
port = 7777
`
	if err := os.WriteFile(filepath.Join(dir, DefaultFileName), []byte(defaultToml), 0o644); err != nil {
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
	if _, err := os.Stat(filepath.Join(dir, FileName)); err != nil {
		t.Errorf("config.toml should have been created: %v", err)
	}
}

func TestWriteTo(t *testing.T) {
	cfg := &Config{
		Server:    ServerConfig{Host: "0.0.0.0", Port: 9090},
		Retrieval: RetrievalConfig{TopK: 3},
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
	if err := os.WriteFile(filepath.Join(dir, DefaultFileName), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, FileName), data, 0o644); err != nil {
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
	if loaded.Retrieval.TopK != 3 {
		t.Errorf("top_k = %d, want 3", loaded.Retrieval.TopK)
	}
}

// TestWriteTo_PreservesUnknownSections locks the contract that WriteTo only
// rewrites the top-level sections owned by Config. Foreign sections (notably
// [eval], consumed by the Python eval harness) must round-trip untouched, and
// shrinking a known slice (a removed [[backends]] entry) must take effect —
// otherwise koanf's path-merge would let the deleted entry survive.
func TestWriteTo_PreservesUnknownSections(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, FileName)

	initial := `
[server]
host = "127.0.0.1"
port = 3000

[[backends]]
name = "a"
base_url = "http://a"
enabled = true

[[backends]]
name = "b"
base_url = "http://b"
enabled = true

[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"
judge_input_price_per_mtok = 3.0
`
	if err := os.WriteFile(path, []byte(initial), 0o644); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	cfg := &Config{
		Server: ServerConfig{Host: "0.0.0.0", Port: 9090},
		Backends: []BackendConfig{
			{Name: "a", BaseURL: "http://a", Enabled: true},
		},
	}
	if err := WriteTo(cfg, path); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}

	k := koanf.New(".")
	if err := k.Load(file.Provider(path), toml.Parser()); err != nil {
		t.Fatalf("reload: %v", err)
	}

	if got := k.String("eval.judge_backend"); got != "remote-claude" {
		t.Errorf("eval.judge_backend = %q, want preserved 'remote-claude'", got)
	}
	if got := k.String("eval.judge_model"); got != "claude-sonnet-4-6" {
		t.Errorf("eval.judge_model = %q, want preserved 'claude-sonnet-4-6'", got)
	}
	if got := k.Float64("eval.judge_input_price_per_mtok"); got != 3.0 {
		t.Errorf("eval.judge_input_price_per_mtok = %v, want preserved 3.0", got)
	}

	if got := k.Int("server.port"); got != 9090 {
		t.Errorf("server.port = %d, want 9090", got)
	}
	if got := k.String("server.host"); got != "0.0.0.0" {
		t.Errorf("server.host = %q, want '0.0.0.0'", got)
	}

	type be struct {
		Name string `koanf:"name"`
	}
	var bs []be
	if err := k.Unmarshal("backends", &bs); err != nil {
		t.Fatalf("unmarshal backends: %v", err)
	}
	if len(bs) != 1 {
		t.Fatalf("backends len = %d, want 1 (deleted entry must not survive); got %#v", len(bs), bs)
	}
	if bs[0].Name != "a" {
		t.Errorf("backends[0].name = %q, want 'a'", bs[0].Name)
	}
}

// TestWriteTo_NoExistingFile covers the bootstrap case: WriteTo to a path that
// does not exist yet should succeed (no preservation work to do) and produce a
// readable file.
func TestWriteTo_NoExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.toml")
	cfg := &Config{Server: ServerConfig{Host: "1.2.3.4", Port: 4242}}
	if err := WriteTo(cfg, path); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	k := koanf.New(".")
	if err := k.Load(file.Provider(path), toml.Parser()); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if got := k.Int("server.port"); got != 4242 {
		t.Errorf("server.port = %d, want 4242", got)
	}
}

// TestLoad_RejectsLegacyKeys locks the hard-break migration: any config still
// using pre-restructure keys ([rag], [api], or pre-vision agent fields) must
// fail loudly with a rename hint instead of silently producing zero defaults
// from koanf.
func TestLoad_RejectsLegacyKeys(t *testing.T) {
	tests := []struct {
		name     string
		toml     string
		mustHint string
	}{
		{"rag.top_k", "[rag]\ntop_k = 5\n", "retrieval.top_k"},
		{"rag.max_context_tokens", "[rag]\nmax_context_tokens = 2048\n", "retrieval.max_context_tokens"},
		{"rag.retrieval_mode", "[rag]\nretrieval_mode = \"dense\"\n", "retrieval.retrieval_mode"},
		{"agent.public_base_url", "[agent]\npublic_base_url = \"http://x\"\n", "agent.vision.public_base_url"},
		{"agent.max_images_per_call", "[agent]\nmax_images_per_call = 4\n", "agent.vision.max_images_per_call"},
		{"agent.max_image_bytes_per_call", "[agent]\nmax_image_bytes_per_call = 1024\n", "agent.vision.max_image_bytes_per_call"},
		{
			"backends.supports_vision",
			"[[backends]]\nname = \"x\"\nbase_url = \"http://x\"\nenabled = true\nsupports_vision = true\n",
			"agent.vision.enabled",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Load(writeTempConfig(t, tt.toml))
			if err == nil {
				t.Fatalf("expected error for legacy key %q, got nil", tt.name)
			}
			if !strings.Contains(err.Error(), tt.mustHint) {
				t.Errorf("error should hint at new key %q, got: %v", tt.mustHint, err)
			}
		})
	}
}

// TestAgentVisionDefaults verifies the defaults applied for the new
// [agent.vision] sub-section so a fresh config (no [agent.vision] block) still
// produces sane runtime values for vision-capable backends. Enabled defaults
// to false: vision is opt-in, since attaching images to a non-vision model
// silently produces garbage.
func TestAgentVisionDefaults(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Agent.Vision.Enabled {
		t.Errorf("agent.vision.enabled = true, want false (opt-in)")
	}
	if cfg.Agent.Vision.MaxImagesPerCall != 6 {
		t.Errorf("agent.vision.max_images_per_call = %d, want 6",
			cfg.Agent.Vision.MaxImagesPerCall)
	}
	if cfg.Agent.Vision.MaxImageBytesPerCall != 65536 {
		t.Errorf("agent.vision.max_image_bytes_per_call = %d, want 65536",
			cfg.Agent.Vision.MaxImageBytesPerCall)
	}
	if cfg.Agent.Vision.PublicBaseURL != "" {
		t.Errorf("agent.vision.public_base_url = %q, want empty",
			cfg.Agent.Vision.PublicBaseURL)
	}
}

func TestAgentVisionDefaultsToInlineImages(t *testing.T) {
	cfg, err := Load(writeTempConfig(t, ``))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Agent.Vision.ImageTransport != ImageTransportInline {
		t.Errorf("agent.vision.image_transport = %q, want %q", cfg.Agent.Vision.ImageTransport, ImageTransportInline)
	}
}

func TestLoadRejectsUnknownImageTransport(t *testing.T) {
	_, err := Load(writeTempConfig(t, "[agent.vision]\nimage_transport = 'ftp'\n"))
	if err == nil || !strings.Contains(err.Error(), "image_transport") {
		t.Fatalf("Load error = %v, want an image_transport error", err)
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

// TestLoadTitleInSearchText checks that the flag is on when a config predates
// it, and that an explicit false is kept.
func TestLoadTitleInSearchText(t *testing.T) {
	for _, tc := range []struct {
		toml string
		want bool
	}{
		{"[server]\nport = 8080\n", true},
		{"[chunking]\ntitle_in_search_text = false\n", false},
		{"[chunking]\ntitle_in_search_text = true\n", true},
	} {
		cfg, err := Load(writeTempConfig(t, tc.toml))
		if err != nil {
			t.Fatalf("Load(%q): %v", tc.toml, err)
		}
		if cfg.Chunking.TitleInSearchText != tc.want {
			t.Errorf("Load(%q): TitleInSearchText = %v, want %v", tc.toml, cfg.Chunking.TitleInSearchText, tc.want)
		}
	}
}

// TestLoadRerankerTitleInText checks that the flag is on when a config
// predates it, and that an explicit false is kept.
func TestLoadRerankerTitleInText(t *testing.T) {
	for _, tc := range []struct {
		toml string
		want bool
	}{
		{"[server]\nport = 8080\n", true},
		{"[reranker]\ntitle_in_text = false\n", false},
		{"[reranker]\ntitle_in_text = true\n", true},
	} {
		cfg, err := Load(writeTempConfig(t, tc.toml))
		if err != nil {
			t.Fatalf("Load(%q): %v", tc.toml, err)
		}
		if cfg.Reranker.TitleInText != tc.want {
			t.Errorf("Load(%q): Reranker.TitleInText = %v, want %v", tc.toml, cfg.Reranker.TitleInText, tc.want)
		}
	}
}
