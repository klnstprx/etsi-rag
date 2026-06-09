package config

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/knadh/koanf/parsers/toml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/providers/structs"
	"github.com/knadh/koanf/v2"
)

const (
	// ConfigFileName is the user-editable config file (gitignored).
	ConfigFileName = "config.toml"
	// DefaultConfigFileName is the tracked template with default values.
	DefaultConfigFileName = "config.default.toml"

	envPrefix    = "APP__"
	envSeparator = "__"
)

// BackendConfig represents an LLM server endpoint.
//
// MaxRetries is the per-call retry budget for transient 5xx responses (e.g.
// llama-server crash). 0 = use llm.DefaultMaxRetries;
// negative is treated as 0. Applies to all clients targeting this backend
// (embed, chat, token counter).
type BackendConfig struct {
	Name       string `koanf:"name"`
	BaseURL    string `koanf:"base_url"`
	APIKey     string `koanf:"api_key"`
	Enabled    bool   `koanf:"enabled"`
	IsLlamaCpp bool   `koanf:"is_llama_cpp"`
	MaxRetries int    `koanf:"max_retries"`
}

// Config holds all application configuration.
//
// Sections are grouped by role:
//   - Infrastructure: server, admin, database, object_store, backends.
//   - Pipeline (chunk → embed → retrieve → answer): chunking, embedding,
//     retrieval, agent, session.
//   - Ingest: crawler.
type Config struct {
	// Infrastructure.
	Server      ServerConfig      `koanf:"server"`
	Admin       AdminConfig       `koanf:"admin"`
	Database    DatabaseConfig    `koanf:"database"`
	ObjectStore ObjectStoreConfig `koanf:"object_store"`
	Backends    []BackendConfig   `koanf:"backends"`

	// Pipeline.
	Chunking  ChunkingConfig  `koanf:"chunking"`
	Embedding EmbeddingConfig `koanf:"embedding"`
	Retrieval RetrievalConfig `koanf:"retrieval"`
	Reranker  RerankerConfig  `koanf:"reranker"`
	Agent     AgentConfig     `koanf:"agent"`
	Session   SessionConfig   `koanf:"session"`

	// Ingest.
	Crawler CrawlerConfig `koanf:"crawler"`
}

// GetBackend returns the first enabled backend with the given name, or nil.
func (c *Config) GetBackend(name string) *BackendConfig {
	for i := range c.Backends {
		if c.Backends[i].Name == name && c.Backends[i].Enabled {
			return &c.Backends[i]
		}
	}
	return nil
}

// MaskAPIKey returns a masked version of an API key showing only the last 4 characters.
func MaskAPIKey(key string) string {
	if key == "" {
		return "(none)"
	}
	if len(key) <= 4 {
		return "••••"
	}
	return "••••" + key[len(key)-4:]
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Host            string        `koanf:"host"`
	Port            int           `koanf:"port"`
	ReadTimeout     int           `koanf:"read_timeout"`     // seconds
	WriteTimeout    int           `koanf:"write_timeout"`    // seconds; should be >= request_timeout + a small grace
	RequestTimeout  int           `koanf:"request_timeout"`  // seconds; per-request handler ceiling, bounds agent tool-calling loops
	IdleTimeout     time.Duration `koanf:"idle_timeout"`     // keep-alive idle
	ShutdownTimeout time.Duration `koanf:"shutdown_timeout"` // graceful shutdown grace
}

// AdminConfig holds admin-UI behaviour settings.
type AdminConfig struct {
	BackendProbeTimeout  time.Duration `koanf:"backend_probe_timeout"`  // per-request timeout for backend health checks
	BackendModelsTimeout time.Duration `koanf:"backend_models_timeout"` // per-request timeout for /models listing
	ChatStreamTimeout    time.Duration `koanf:"chat_stream_timeout"`    // max lifetime of an admin chat turn goroutine
}

// EmbeddingConfig holds embedding stage settings.
type EmbeddingConfig struct {
	Backend   string `koanf:"backend"`
	Model     string `koanf:"model"`
	Dimension int    `koanf:"dimension"`
	MaxTokens int    `koanf:"max_tokens"`
	BatchSize int    `koanf:"batch_size"`
}

// AgentConfig holds agent/chat stage settings.
type AgentConfig struct {
	Backend             string            `koanf:"backend"`
	Model               string            `koanf:"model"`
	MaxTokens           int               `koanf:"max_tokens"`
	Temperature         float64           `koanf:"temperature"`
	ContextWindow       int               `koanf:"context_window"`         // total context window (auto-detected for llama.cpp)
	MaxIterations       int               `koanf:"max_iterations"`         // max tool-calling iterations
	ToolOutputMaxTokens int               `koanf:"tool_output_max_tokens"` // max tokens allowed per tool result; 0 = auto (context_window/4)
	SystemPrompt        string            `koanf:"system_prompt"`          // empty = use prompts/agent.md
	Vision              AgentVisionConfig `koanf:"vision"`
}

// AgentVisionConfig holds image-attachment settings for the agent stage.
// Whether the configured agent.model actually supports vision is asserted by
// the operator via Enabled — the runtime cannot tell across all backends
// (llama.cpp's /props is a hint, OpenAI-compatible providers expose nothing).
type AgentVisionConfig struct {
	Enabled              bool   `koanf:"enabled"`                  // operator asserts the configured agent.model has vision; gates image attachment
	MaxImagesPerCall     int    `koanf:"max_images_per_call"`      // max image artifacts attached per tool call; 0 = default 6
	MaxImageBytesPerCall int64  `koanf:"max_image_bytes_per_call"` // max bytes of image_url URLs (text length, not image data) per tool call; 0 = default 65536
	PublicBaseURL        string `koanf:"public_base_url"`          // absolute base URL prefixed onto /v1/artifacts/... paths so the LLM backend can fetch them; empty disables image attachment
}

// DatabaseConfig holds SQLite database settings.
type DatabaseConfig struct {
	Path string `koanf:"path"`
}

// RetrievalConfig holds retrieval-pipeline settings (top-k, context budget, mode).
type RetrievalConfig struct {
	TopK             int `koanf:"top_k"`
	MaxContextTokens int `koanf:"max_context_tokens"`
	// RetrievalMode selects how chunks are scored:
	//   "dense"  — cosine similarity over query embedding (default).
	//   "hybrid" — dense + BM25 keyword search merged via reciprocal-rank
	//              fusion. See ADR-008.
	RetrievalMode string `koanf:"retrieval_mode"`
}

// RerankerConfig holds reranker stage settings. The reranker is opt-in
// (Enabled) and llama-cpp only — the /rerank endpoint is a llama.cpp/Jina
// extension, not part of the OpenAI-compatible surface, so non-llama-cpp
// backends are rejected at config-load time. See ADR-009.
type RerankerConfig struct {
	Enabled        bool          `koanf:"enabled"`
	Backend        string        `koanf:"backend"`
	Model          string        `koanf:"model"`
	TopN           int           `koanf:"top_n"`           // candidate pool size; must be >= retrieval.top_k
	MaxTokens      int           `koanf:"max_tokens"`      // 0 = auto-detect via /props
	ReservedTokens int           `koanf:"reserved_tokens"` // BOS/EOS/sep budget reserved before truncation
	Timeout        time.Duration `koanf:"timeout"`
}

// ChunkingConfig holds document chunking settings.
type ChunkingConfig struct {
	MaxChunkSize int      `koanf:"max_chunk_size"`
	ChunkOverlap int      `koanf:"chunk_overlap"`
	MinChunkSize int      `koanf:"min_chunk_size"`
	Separators   []string `koanf:"separators"`
}

// SessionConfig holds session management settings.
type SessionConfig struct {
	MaxMessages int `koanf:"max_messages"`
	TTLSeconds  int `koanf:"ttl_seconds"`
}

// CrawlerConfig holds web crawler settings.
type CrawlerConfig struct {
	RequestDelay        time.Duration `koanf:"request_delay"`
	RequestTimeout      time.Duration `koanf:"request_timeout"`
	ProgressLogInterval time.Duration `koanf:"progress_log_interval"` // how often to log crawl progress
	MaxDepth            int           `koanf:"max_depth"`
	MaxPages            int           `koanf:"max_pages"`
	Concurrency         int           `koanf:"concurrency"`
	MaxBodySize         int64         `koanf:"max_body_size"`      // max response body in bytes, 0 = unlimited
	MinContentLen       int           `koanf:"min_content_length"` // skip pages with less content (chars), 0 = disabled
	EmbedWorkers        int           `koanf:"embed_workers"`      // concurrent embed stage workers
	ParseWorkers        int           `koanf:"parse_workers"`      // concurrent parse stage workers
	MaxEmbedFails       int           `koanf:"max_embed_fails"`    // consecutive embed failures before aborting crawl
	JSPatterns          []string      `koanf:"js_patterns"`        // URL path prefixes requiring JS rendering
	RejectionPatterns   []string      `koanf:"rejection_patterns"` // case-insensitive substrings that mark a page as garbage (soft 404s, stubs, CMS placeholders)
	StripQueryParams    []string      `koanf:"strip_query_params"` // case-insensitive query-param names to strip during URL normalization (tracking, session tokens)
	PDF                 PDFConfig     `koanf:"pdf"`
}

// PDFConfig holds PDF text extraction settings.
type PDFConfig struct {
	HeadingPatterns []string `koanf:"heading_patterns"` // substring patterns for known section headers
	RenderDPI       int      `koanf:"render_dpi"`       // pdftoppm -r <dpi> for page renders, 0 disables image rendering
}

// ObjectStoreConfig holds S3-compatible object storage settings (RustFS).
type ObjectStoreConfig struct {
	Endpoint  string `koanf:"endpoint"`
	Bucket    string `koanf:"bucket"`
	AccessKey string `koanf:"access_key"`
	SecretKey string `koanf:"secret_key"`
	Region    string `koanf:"region"`
}

// ConfigPath returns the full path to the user config file within a config directory.
func ConfigPath(configDir string) string {
	return configDir + "/" + ConfigFileName
}

// DefaultConfigPath returns the full path to the default config template within a config directory.
func DefaultConfigPath(configDir string) string {
	return configDir + "/" + DefaultConfigFileName
}

// ensureConfigFile copies the default config to the user config path if it does not exist.
func ensureConfigFile(configDir string) error {
	cfgPath := ConfigPath(configDir)
	if _, err := os.Stat(cfgPath); err == nil {
		return nil // already exists
	}

	defaultPath := DefaultConfigPath(configDir)
	src, err := os.Open(defaultPath)
	if err != nil {
		return fmt.Errorf("opening %s: %w", DefaultConfigFileName, err)
	}
	defer src.Close()

	dst, err := os.Create(cfgPath)
	if err != nil {
		return fmt.Errorf("creating %s: %w", ConfigFileName, err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		return fmt.Errorf("copying default config: %w", err)
	}
	return nil
}

// Load reads configuration from files and environment variables.
// It ensures config.toml exists (copying from config.default.toml if needed),
// then loads config.toml → APP__* env vars.
func Load(configDir string) (*Config, error) {
	if err := ensureConfigFile(configDir); err != nil {
		return nil, fmt.Errorf("ensuring config file: %w", err)
	}

	k := koanf.New(".")

	// Load user config
	cfgPath := ConfigPath(configDir)
	if err := k.Load(file.Provider(cfgPath), toml.Parser()); err != nil {
		return nil, fmt.Errorf("loading %s: %w", ConfigFileName, err)
	}

	// Load environment variables with APP__ prefix
	if err := k.Load(env.Provider(envPrefix, ".", func(s string) string {
		return strings.ReplaceAll(
			strings.ToLower(strings.TrimPrefix(s, envPrefix)),
			envSeparator,
			".",
		)
	}), nil); err != nil {
		return nil, fmt.Errorf("loading env vars: %w", err)
	}

	if err := checkLegacyKeys(k); err != nil {
		return nil, err
	}

	var cfg Config
	if err := k.Unmarshal("", &cfg); err != nil {
		return nil, err
	}

	// Apply defaults for optional fields
	cfg.applyDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// checkLegacyKeys returns a friendly error if the loaded config still uses keys
// from the pre-restructure layout. The migration is a hard break: rather than
// silently producing zero defaults via koanf when a [rag] or [api] section is
// present, fail loud and tell the operator exactly what to rename.
func checkLegacyKeys(k *koanf.Koanf) error {
	legacy := []struct {
		old, new string
	}{
		{"rag.top_k", "retrieval.top_k"},
		{"rag.max_context_tokens", "retrieval.max_context_tokens"},
		{"rag.retrieval_mode", "retrieval.retrieval_mode"},
		{"agent.public_base_url", "agent.vision.public_base_url"},
		{"agent.max_images_per_call", "agent.vision.max_images_per_call"},
		{"agent.max_image_bytes_per_call", "agent.vision.max_image_bytes_per_call"},
	}
	var found []string
	for _, l := range legacy {
		if k.Exists(l.old) {
			found = append(found, fmt.Sprintf("  %s → %s", l.old, l.new))
		}
	}
	// Per-backend supports_vision moved to the agent stage when the design
	// shifted from "backend can serve vision-capable models" to "the
	// configured agent.model is vision-capable". Walk the raw backends slice
	// rather than guessing indexes; koanf paths for repeated TOML arrays use
	// numeric indexes (backends.0.supports_vision).
	if rawBackends, ok := k.Raw()["backends"].([]any); ok {
		for i, raw := range rawBackends {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if _, exists := m["supports_vision"]; exists {
				found = append(found, fmt.Sprintf(
					"  backends[%d].supports_vision → agent.vision.enabled (moved to agent stage; see ADR-006)", i))
			}
		}
	}
	if len(found) == 0 {
		return nil
	}
	return fmt.Errorf(
		"config uses legacy keys; rename the following in your %s and retry:\n%s",
		ConfigFileName, strings.Join(found, "\n"),
	)
}

// Validate enforces cross-section invariants that applyDefaults can't express.
// Called by Load after defaults are applied; also called by the admin UI before
// persisting a section edit, so a bad submission rejects without touching disk.
func (c *Config) Validate() error {
	if c.Chunking.MaxChunkSize > c.Retrieval.MaxContextTokens {
		return fmt.Errorf(
			"chunking.max_chunk_size (%d) must not exceed retrieval.max_context_tokens (%d): "+
				"a single chunk that doesn't fit the retrieval budget would starve the agent",
			c.Chunking.MaxChunkSize, c.Retrieval.MaxContextTokens,
		)
	}
	switch c.Retrieval.RetrievalMode {
	case "dense", "hybrid":
		// ok
	default:
		return fmt.Errorf(
			"retrieval.retrieval_mode = %q is not supported; use \"dense\" or \"hybrid\"",
			c.Retrieval.RetrievalMode,
		)
	}
	if c.Reranker.Enabled {
		if c.Reranker.Backend == "" {
			return fmt.Errorf("reranker.enabled = true requires reranker.backend to be set")
		}
		if c.Reranker.Model == "" {
			return fmt.Errorf("reranker.enabled = true requires reranker.model to be set")
		}
		b := c.GetBackend(c.Reranker.Backend)
		if b == nil {
			return fmt.Errorf(
				"reranker.backend = %q does not match any enabled [[backends]] entry",
				c.Reranker.Backend,
			)
		}
		if !b.IsLlamaCpp {
			return fmt.Errorf(
				"reranker.backend = %q is not a llama.cpp backend; the /rerank endpoint is "+
					"a llama.cpp extension and not part of the OpenAI-compatible surface",
				c.Reranker.Backend,
			)
		}
		if c.Reranker.TopN < c.Retrieval.TopK {
			return fmt.Errorf(
				"reranker.top_n (%d) must be >= retrieval.top_k (%d): "+
					"the candidate pool needs at least as many docs as the agent will see",
				c.Reranker.TopN, c.Retrieval.TopK,
			)
		}
	}
	return nil
}

// applyDefaults sets default values for fields not specified in config.
func (c *Config) applyDefaults() {
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 15
	}
	if c.Server.RequestTimeout == 0 {
		c.Server.RequestTimeout = 120
	}
	// WriteTimeout must outlast RequestTimeout so the handler timeout fires
	// first and produces a clean 503 instead of a connection reset.
	if c.Server.WriteTimeout == 0 {
		c.Server.WriteTimeout = c.Server.RequestTimeout + 180
	}
	if c.Server.IdleTimeout == 0 {
		c.Server.IdleTimeout = 60 * time.Second
	}
	if c.Server.ShutdownTimeout == 0 {
		c.Server.ShutdownTimeout = 30 * time.Second
	}

	if c.Admin.BackendProbeTimeout == 0 {
		c.Admin.BackendProbeTimeout = 3 * time.Second
	}
	if c.Admin.BackendModelsTimeout == 0 {
		c.Admin.BackendModelsTimeout = 5 * time.Second
	}
	if c.Admin.ChatStreamTimeout == 0 {
		c.Admin.ChatStreamTimeout = 5 * time.Minute
	}

	if c.Agent.MaxTokens == 0 {
		c.Agent.MaxTokens = 1024
	}
	if c.Agent.Temperature == 0 {
		c.Agent.Temperature = 0.7
	}
	if c.Agent.MaxIterations == 0 {
		c.Agent.MaxIterations = 5
	}
	if c.Agent.Vision.MaxImagesPerCall == 0 {
		c.Agent.Vision.MaxImagesPerCall = 6
	}
	if c.Agent.Vision.MaxImageBytesPerCall == 0 {
		c.Agent.Vision.MaxImageBytesPerCall = 65536
	}

	if c.Retrieval.TopK == 0 {
		c.Retrieval.TopK = 10
	}
	if c.Retrieval.MaxContextTokens == 0 {
		c.Retrieval.MaxContextTokens = 4092
	}
	if c.Retrieval.RetrievalMode == "" {
		c.Retrieval.RetrievalMode = "dense"
	}

	if c.Reranker.TopN == 0 {
		c.Reranker.TopN = 30
	}
	if c.Reranker.ReservedTokens == 0 {
		c.Reranker.ReservedTokens = 16
	}
	if c.Reranker.Timeout == 0 {
		c.Reranker.Timeout = 30 * time.Second
	}

	if c.Session.MaxMessages == 0 {
		c.Session.MaxMessages = 10
	}
	if c.Session.TTLSeconds == 0 {
		c.Session.TTLSeconds = 3600
	}

	if c.Embedding.MaxTokens == 0 {
		c.Embedding.MaxTokens = 512
	}
	if c.Embedding.BatchSize == 0 {
		c.Embedding.BatchSize = 2
	}

	if c.Chunking.MaxChunkSize == 0 {
		c.Chunking.MaxChunkSize = 256
	}
	if c.Chunking.ChunkOverlap == 0 {
		c.Chunking.ChunkOverlap = 32
	}
	if c.Chunking.MinChunkSize == 0 {
		c.Chunking.MinChunkSize = 64
	}
	if len(c.Chunking.Separators) == 0 {
		c.Chunking.Separators = []string{"\n\n", "\n", ". ", " ", ""}
	}

	if c.Crawler.RequestDelay == 0 {
		c.Crawler.RequestDelay = 500 * time.Millisecond
	}
	if c.Crawler.RequestTimeout == 0 {
		c.Crawler.RequestTimeout = 30 * time.Second
	}
	if c.Crawler.ProgressLogInterval == 0 {
		c.Crawler.ProgressLogInterval = 30 * time.Second
	}
	if c.Crawler.MaxDepth == 0 {
		c.Crawler.MaxDepth = 10
	}
	if c.Crawler.MaxPages == 0 {
		c.Crawler.MaxPages = 5000
	}
	if c.Crawler.Concurrency == 0 {
		c.Crawler.Concurrency = 4
	}
	if c.Crawler.MaxBodySize == 0 {
		c.Crawler.MaxBodySize = 10 * 1024 * 1024 // 10 MB
	}
	if c.Crawler.EmbedWorkers == 0 {
		c.Crawler.EmbedWorkers = 1
	}
	if c.Crawler.ParseWorkers == 0 {
		c.Crawler.ParseWorkers = 4
	}
	if c.Crawler.MinContentLen == 0 {
		c.Crawler.MinContentLen = 200
	}
	if c.Crawler.MaxEmbedFails == 0 {
		c.Crawler.MaxEmbedFails = 3
	}
	if len(c.Crawler.RejectionPatterns) == 0 {
		c.Crawler.RejectionPatterns = []string{
			"página no encontrada",       // soft 404
			"error 404",                  // soft 404
			"página en construcción",     // stub
			"sitio en construcción",      // stub
			"sin descripción disponible", // CMS placeholder
		}
	}
	if len(c.Crawler.StripQueryParams) == 0 {
		c.Crawler.StripQueryParams = []string{
			"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
			"fbclid", "gclid", "_ga", "ref", "lang",
		}
	}
	if len(c.Crawler.PDF.HeadingPatterns) == 0 {
		c.Crawler.PDF.HeadingPatterns = []string{
			"DESCRIPCION", "OBJETIVOS", "CONTENIDOS", "METODOLOGIA",
			"EVALUACION", "COMPETENCIAS", "TEMARIO", "BIBLIOGRAFIA",
			"ACTIVIDADES FORMATIVAS", "SISTEMA DE EVALUACION",
			"RESULTADOS DE APRENDIZAJE", "EQUIPO DOCENTE",
			"RECOMENDACIONES", "DISTRIBUCION DEL TRABAJO",
		}
	}
	if c.Crawler.PDF.RenderDPI == 0 {
		c.Crawler.PDF.RenderDPI = 120
	}

	if c.ObjectStore.Region == "" {
		c.ObjectStore.Region = "us-east-1"
	}
}

// WriteTo marshals the config to TOML and writes it atomically to the given path.
// Uses koanf + structs provider so keys always match the koanf tags used for reading.
//
// Top-level sections not declared on Config (e.g. [eval], consumed by the
// Python eval harness) are preserved: WriteTo loads the existing file first,
// drops the keys this struct owns so stale slice entries don't leak through
// koanf's path-merge, and overlays the typed struct on top.
func WriteTo(cfg *Config, path string) error {
	merged := koanf.New(".")
	if _, err := os.Stat(path); err == nil {
		if err := merged.Load(file.Provider(path), toml.Parser()); err != nil {
			return fmt.Errorf("loading existing config: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("stat config: %w", err)
	}

	owned := koanf.New(".")
	if err := owned.Load(structs.Provider(cfg, "koanf"), nil); err != nil {
		return fmt.Errorf("loading config into koanf: %w", err)
	}
	for k := range owned.Raw() {
		merged.Delete(k)
	}
	if err := merged.Merge(owned); err != nil {
		return fmt.Errorf("merging config: %w", err)
	}

	data, err := merged.Marshal(toml.Parser())
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("writing temp config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("renaming config: %w", err)
	}
	return nil
}
