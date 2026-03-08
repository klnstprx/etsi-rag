package config

import (
	"etsi-rag/internal/prompts"
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
	envRunMode   = "RUN_MODE"
	defaultMode  = "development"
)

// BackendConfig represents an LLM server endpoint.
type BackendConfig struct {
	Name       string `koanf:"name"`
	BaseURL    string `koanf:"base_url"`
	APIKey     string `koanf:"api_key"`
	Enabled    bool   `koanf:"enabled"`
	IsLlamaCpp bool   `koanf:"is_llama_cpp"`
}

// Config holds all application configuration.
type Config struct {
	Server      ServerConfig      `koanf:"server"`
	Backends    []BackendConfig   `koanf:"backends"`
	Embedding   EmbeddingConfig   `koanf:"embedding"`
	Chat        ChatConfig        `koanf:"chat"`
	Agent       AgentConfig       `koanf:"agent"`
	Database    DatabaseConfig    `koanf:"database"`
	RAG         RAGConfig         `koanf:"rag"`
	Chunking    ChunkingConfig    `koanf:"chunking"`
	Session     SessionConfig     `koanf:"session"`
	Crawler     CrawlerConfig     `koanf:"crawler"`
	ObjectStore ObjectStoreConfig `koanf:"object_store"`
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
	Host         string `koanf:"host"`
	Port         int    `koanf:"port"`
	ReadTimeout  int    `koanf:"read_timeout"`
	WriteTimeout int    `koanf:"write_timeout"`
}

// EmbeddingConfig holds embedding stage settings.
type EmbeddingConfig struct {
	Backend   string `koanf:"backend"`
	Model     string `koanf:"model"`
	Dimension int    `koanf:"dimension"`
	MaxTokens int    `koanf:"max_tokens"`
}

// ChatConfig holds chat stage settings.
type ChatConfig struct {
	Backend     string  `koanf:"backend"`
	Model       string  `koanf:"model"`
	MaxTokens   int     `koanf:"max_tokens"`
	Temperature float64 `koanf:"temperature"`
}

// AgentConfig holds agent stage settings.
type AgentConfig struct {
	Backend     string  `koanf:"backend"`
	Model       string  `koanf:"model"`
	MaxTokens   int     `koanf:"max_tokens"`
	Temperature float64 `koanf:"temperature"`
}

// DatabaseConfig holds SQLite database settings.
type DatabaseConfig struct {
	Path string `koanf:"path"`
}

// RAGConfig holds RAG pipeline settings.
type RAGConfig struct {
	TopK                int     `koanf:"top_k"`
	SimilarityThreshold float64 `koanf:"similarity_threshold"`
	MaxContextTokens    int     `koanf:"max_context_tokens"`
	SystemPromptES      string  `koanf:"system_prompt_es"`
	SystemPromptEN      string  `koanf:"system_prompt_en"`
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
	RequestDelay   time.Duration `koanf:"request_delay"`
	RequestTimeout time.Duration `koanf:"request_timeout"`
	MaxDepth       int           `koanf:"max_depth"`
	MaxPages       int           `koanf:"max_pages"`
	Concurrency    int           `koanf:"concurrency"`
	MaxBodySize    int64         `koanf:"max_body_size"` // max response body in bytes, 0 = unlimited
	EmbedWorkers   int           `koanf:"embed_workers"` // concurrent embed stage workers
	ParseWorkers   int           `koanf:"parse_workers"`   // concurrent parse stage workers
	MaxEmbedFails  int           `koanf:"max_embed_fails"` // consecutive embed failures before aborting crawl
	JSPatterns     []string      `koanf:"js_patterns"`   // URL path prefixes requiring JS rendering
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
// then loads config.toml → {RUN_MODE}.toml → APP__* env vars.
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

	// Load environment-specific config if RUN_MODE is set
	runMode := os.Getenv(envRunMode)
	if runMode == "" {
		runMode = defaultMode
	}
	modePath := configDir + "/" + runMode + ".toml"
	if _, err := os.Stat(modePath); err == nil {
		if err := k.Load(file.Provider(modePath), toml.Parser()); err != nil {
			return nil, fmt.Errorf("loading %s: %w", modePath, err)
		}
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

	var cfg Config
	if err := k.Unmarshal("", &cfg); err != nil {
		return nil, err
	}

	// Apply defaults for optional fields
	cfg.applyDefaults()

	return &cfg, nil
}

// applyDefaults sets default values for fields not specified in config.
func (c *Config) applyDefaults() {
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 15
	}
	if c.Server.WriteTimeout == 0 {
		c.Server.WriteTimeout = 120
	}

	if c.Chat.MaxTokens == 0 {
		c.Chat.MaxTokens = 1024
	}
	if c.Chat.Temperature == 0 {
		c.Chat.Temperature = 0.7
	}
	if c.Agent.MaxTokens == 0 {
		c.Agent.MaxTokens = 1024
	}
	if c.Agent.Temperature == 0 {
		c.Agent.Temperature = 0.7
	}

	if c.RAG.TopK == 0 {
		c.RAG.TopK = 5
	}
	if c.RAG.MaxContextTokens == 0 {
		c.RAG.MaxContextTokens = 2048
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
	if c.Crawler.MaxEmbedFails == 0 {
		c.Crawler.MaxEmbedFails = 3
	}

	if c.ObjectStore.Region == "" {
		c.ObjectStore.Region = "us-east-1"
	}

	if c.RAG.SystemPromptES == "" {
		c.RAG.SystemPromptES = prompts.RAGEs
	}
	if c.RAG.SystemPromptEN == "" {
		c.RAG.SystemPromptEN = prompts.RAGEn
	}
}

// WriteTo marshals the config to TOML and writes it atomically to the given path.
// Uses koanf + structs provider so keys always match the koanf tags used for reading.
func WriteTo(cfg *Config, path string) error {
	k := koanf.New(".")
	if err := k.Load(structs.Provider(cfg, "koanf"), nil); err != nil {
		return fmt.Errorf("loading config into koanf: %w", err)
	}

	data, err := k.Marshal(toml.Parser())
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
