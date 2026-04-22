package admin

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/config"
	"etsi-rag/internal/prompts"
)

// ConfigPage handles GET /admin/config.
func (h *Handler) ConfigPage(w http.ResponseWriter, r *http.Request) {
	sections := configToSections(h.config)

	page := templates.ConfigPageTempl(sections)
	content := templates.ConfigContent(sections)
	render(w, r, page, content)
}

// ConfigSave handles POST /admin/config/{section}.
func (h *Handler) ConfigSave(w http.ResponseWriter, r *http.Request) {
	section := chi.URLParam(r, "section")
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	var dimensionChanged bool
	switch section {
	case "server":
		h.config.Server.Host = r.FormValue("host")
		h.config.Server.Port = formInt(r, "port", h.config.Server.Port)
		h.config.Server.ReadTimeout = formInt(r, "read_timeout", h.config.Server.ReadTimeout)
		h.config.Server.WriteTimeout = formInt(r, "write_timeout", h.config.Server.WriteTimeout)
	case "rag":
		h.config.RAG.TopK = formInt(r, "top_k", h.config.RAG.TopK)
		h.config.RAG.MaxContextTokens = formInt(r, "max_context_tokens", h.config.RAG.MaxContextTokens)
	case "chunking":
		h.config.Chunking.MaxChunkSize = formInt(r, "max_chunk_size", h.config.Chunking.MaxChunkSize)
		h.config.Chunking.ChunkOverlap = formInt(r, "chunk_overlap", h.config.Chunking.ChunkOverlap)
		h.config.Chunking.MinChunkSize = formInt(r, "min_chunk_size", h.config.Chunking.MinChunkSize)
		if v := r.FormValue("separators"); v != "" {
			h.config.Chunking.Separators = strings.Split(v, ",")
		}
	case "session":
		h.config.Session.MaxMessages = formInt(r, "max_messages", h.config.Session.MaxMessages)
		h.config.Session.TTLSeconds = formInt(r, "ttl_seconds", h.config.Session.TTLSeconds)
	case "embedding":
		oldDim := h.config.Embedding.Dimension
		h.config.Embedding.Backend = r.FormValue("backend")
		h.config.Embedding.Model = r.FormValue("model")
		h.config.Embedding.Dimension = formInt(r, "dimension", h.config.Embedding.Dimension)
		h.config.Embedding.MaxTokens = formInt(r, "max_tokens", h.config.Embedding.MaxTokens)
		h.config.Embedding.BatchSize = formInt(r, "batch_size", h.config.Embedding.BatchSize)
		dimensionChanged = oldDim != 0 && h.config.Embedding.Dimension != oldDim
	case "agent":
		h.config.Agent.Backend = r.FormValue("backend")
		h.config.Agent.Model = r.FormValue("model")
		h.config.Agent.MaxTokens = formInt(r, "max_tokens", h.config.Agent.MaxTokens)
		h.config.Agent.Temperature = formFloat(r, "temperature", h.config.Agent.Temperature)
		h.config.Agent.ContextWindow = formInt(r, "context_window", h.config.Agent.ContextWindow)
		h.config.Agent.MaxIterations = formInt(r, "max_iterations", h.config.Agent.MaxIterations)
		h.config.Agent.ToolOutputMaxTokens = formInt(r, "tool_output_max_tokens", h.config.Agent.ToolOutputMaxTokens)
		submitted := strings.TrimSpace(r.FormValue("system_prompt"))
		if submitted == "" || submitted == strings.TrimSpace(prompts.Agent) {
			h.config.Agent.SystemPrompt = ""
		} else {
			h.config.Agent.SystemPrompt = submitted
		}
	case "api":
		h.config.API.Passthrough = r.FormValue("passthrough") == "true"
	case "database":
		h.config.Database.Path = r.FormValue("path")
	case "crawler":
		h.config.Crawler.MaxDepth = formInt(r, "max_depth", h.config.Crawler.MaxDepth)
		h.config.Crawler.MaxPages = formInt(r, "max_pages", h.config.Crawler.MaxPages)
		h.config.Crawler.Concurrency = formInt(r, "concurrency", h.config.Crawler.Concurrency)
		h.config.Crawler.MaxBodySize = int64(formInt(r, "max_body_size", int(h.config.Crawler.MaxBodySize)))
		h.config.Crawler.EmbedWorkers = formInt(r, "embed_workers", h.config.Crawler.EmbedWorkers)
		h.config.Crawler.ParseWorkers = formInt(r, "parse_workers", h.config.Crawler.ParseWorkers)
		h.config.Crawler.MinContentLen = formInt(r, "min_content_len", h.config.Crawler.MinContentLen)
		h.config.Crawler.MaxEmbedFails = formInt(r, "max_embed_fails", h.config.Crawler.MaxEmbedFails)
		h.config.Crawler.RequestDelay = time.Duration(formInt(r, "request_delay_ms", int(h.config.Crawler.RequestDelay/time.Millisecond))) * time.Millisecond
		h.config.Crawler.RequestTimeout = time.Duration(formInt(r, "request_timeout_s", int(h.config.Crawler.RequestTimeout/time.Second))) * time.Second
		if v := r.FormValue("js_patterns"); v != "" {
			h.config.Crawler.JSPatterns = splitTrimCSV(v)
		} else {
			h.config.Crawler.JSPatterns = nil
		}
		if v := r.FormValue("pdf_heading_patterns"); v != "" {
			h.config.Crawler.PDF.HeadingPatterns = splitTrimCSV(v)
		} else {
			h.config.Crawler.PDF.HeadingPatterns = nil
		}
		if v := r.FormValue("rejection_patterns"); v != "" {
			h.config.Crawler.RejectionPatterns = splitTrimCSV(v)
		} else {
			h.config.Crawler.RejectionPatterns = nil
		}
		if v := r.FormValue("strip_query_params"); v != "" {
			h.config.Crawler.StripQueryParams = splitTrimCSV(v)
		} else {
			h.config.Crawler.StripQueryParams = nil
		}
	case "object_store":
		h.config.ObjectStore.Endpoint = r.FormValue("endpoint")
		h.config.ObjectStore.Bucket = r.FormValue("bucket")
		h.config.ObjectStore.AccessKey = r.FormValue("access_key")
		h.config.ObjectStore.SecretKey = r.FormValue("secret_key")
		h.config.ObjectStore.Region = r.FormValue("region")
	default:
		http.Error(w, "unknown section", http.StatusBadRequest)
		return
	}

	if err := config.WriteTo(h.config, h.configPath); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		sections := configToSections(h.config)
		templates.ConfigContent(sections).Render(r.Context(), w)
		templates.Toast("Error saving config: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	h.fireConfigSave()
	h.logger.Info().Str("section", section).Msg("config saved")

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	sections := configToSections(h.config)
	templates.ConfigContent(sections).Render(r.Context(), w)
	templates.Toast(fmt.Sprintf("Section '%s' saved", section), "success").Render(r.Context(), w)
	if dimensionChanged {
		templates.Toast("Embedding dimension changed — existing documents should be reprocessed", "warning").Render(r.Context(), w)
	}
}

func configToSections(cfg *config.Config) []templates.ConfigSection {
	backendNames := make([]string, len(cfg.Backends))
	for i, b := range cfg.Backends {
		backendNames[i] = b.Name
	}

	return []templates.ConfigSection{
		{
			Name: "server",
			Fields: []templates.ConfigField{
				{Key: "host", Label: "Host", Value: cfg.Server.Host, Type: "text"},
				{Key: "port", Label: "Port", Value: strconv.Itoa(cfg.Server.Port), Type: "number"},
				{Key: "read_timeout", Label: "Read Timeout (s)", Value: strconv.Itoa(cfg.Server.ReadTimeout), Type: "number"},
				{Key: "write_timeout", Label: "Write Timeout (s)", Value: strconv.Itoa(cfg.Server.WriteTimeout), Type: "number"},
			},
		},
		{
			Name: "embedding",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Embedding.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Embedding.Model, Type: "model", Options: backendNames},
				{Key: "dimension", Label: "Dimension", Value: strconv.Itoa(cfg.Embedding.Dimension), Type: "number"},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Embedding.MaxTokens), Type: "number"},
				{Key: "batch_size", Label: "Batch Size", Value: strconv.Itoa(cfg.Embedding.BatchSize), Type: "number"},
			},
		},
		{
			Name: "agent",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Agent.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Agent.Model, Type: "model", Options: backendNames},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Agent.MaxTokens), Type: "number"},
				{Key: "temperature", Label: "Temperature", Value: fmt.Sprintf("%.2f", cfg.Agent.Temperature), Type: "text"},
				{Key: "context_window", Label: "Context Window (auto-detected)", Value: strconv.Itoa(cfg.Agent.ContextWindow), Type: "number"},
				{Key: "max_iterations", Label: "Max Tool Iterations", Value: strconv.Itoa(cfg.Agent.MaxIterations), Type: "number"},
				{Key: "tool_output_max_tokens", Label: "Tool Output Max Tokens (0 = auto, context_window/4)", Value: strconv.Itoa(cfg.Agent.ToolOutputMaxTokens), Type: "number"},
				{Key: "system_prompt", Label: "System Prompt (clear to restore default)", Value: effectivePrompt(cfg), Type: "textarea", Hint: "Template variables: {{.DocumentCount}}, {{.ChunkCount}}"},
			},
		},
		{
			Name: "rag",
			Fields: []templates.ConfigField{
				{Key: "top_k", Label: "Top K", Value: strconv.Itoa(cfg.RAG.TopK), Type: "number"},
				{Key: "max_context_tokens", Label: "Max Context Tokens", Value: strconv.Itoa(cfg.RAG.MaxContextTokens), Type: "number"},
			},
		},
		{
			Name: "api",
			Fields: []templates.ConfigField{
				{Key: "passthrough", Label: "Passthrough Mode", Value: fmt.Sprintf("%t", cfg.API.Passthrough), Type: "text"},
			},
		},
		{
			Name: "chunking",
			Fields: []templates.ConfigField{
				{Key: "max_chunk_size", Label: "Max Chunk Size (tokens)", Value: strconv.Itoa(cfg.Chunking.MaxChunkSize), Type: "number"},
				{Key: "chunk_overlap", Label: "Chunk Overlap (tokens)", Value: strconv.Itoa(cfg.Chunking.ChunkOverlap), Type: "number"},
				{Key: "min_chunk_size", Label: "Min Chunk Size (tokens)", Value: strconv.Itoa(cfg.Chunking.MinChunkSize), Type: "number"},
				{Key: "separators", Label: "Separators (comma-separated)", Value: strings.Join(cfg.Chunking.Separators, ","), Type: "text"},
			},
		},
		{
			Name: "session",
			Fields: []templates.ConfigField{
				{Key: "max_messages", Label: "Max Messages", Value: strconv.Itoa(cfg.Session.MaxMessages), Type: "number"},
				{Key: "ttl_seconds", Label: "TTL (seconds)", Value: strconv.Itoa(cfg.Session.TTLSeconds), Type: "number"},
			},
		},
		{
			Name: "database",
			Fields: []templates.ConfigField{
				{Key: "path", Label: "Path", Value: cfg.Database.Path, Type: "text"},
			},
		},
		{
			Name: "crawler",
			Fields: []templates.ConfigField{
				{Key: "request_delay_ms", Label: "Request Delay (ms)", Value: strconv.FormatInt(cfg.Crawler.RequestDelay.Milliseconds(), 10), Type: "number"},
				{Key: "request_timeout_s", Label: "Request Timeout (s)", Value: strconv.FormatInt(int64(cfg.Crawler.RequestTimeout/time.Second), 10), Type: "number"},
				{Key: "max_depth", Label: "Max Depth", Value: strconv.Itoa(cfg.Crawler.MaxDepth), Type: "number"},
				{Key: "max_pages", Label: "Max Pages", Value: strconv.Itoa(cfg.Crawler.MaxPages), Type: "number"},
				{Key: "concurrency", Label: "Concurrency", Value: strconv.Itoa(cfg.Crawler.Concurrency), Type: "number"},
				{Key: "max_body_size", Label: "Max Body Size (bytes)", Value: strconv.FormatInt(cfg.Crawler.MaxBodySize, 10), Type: "number"},
				{Key: "embed_workers", Label: "Embed Workers", Value: strconv.Itoa(cfg.Crawler.EmbedWorkers), Type: "number"},
				{Key: "parse_workers", Label: "Parse Workers", Value: strconv.Itoa(cfg.Crawler.ParseWorkers), Type: "number"},
				{Key: "min_content_len", Label: "Min Content Length (chars)", Value: strconv.Itoa(cfg.Crawler.MinContentLen), Type: "number"},
				{Key: "max_embed_fails", Label: "Max Embed Failures", Value: strconv.Itoa(cfg.Crawler.MaxEmbedFails), Type: "number"},
				{Key: "js_patterns", Label: "JS Patterns (comma-separated)", Value: strings.Join(cfg.Crawler.JSPatterns, ", "), Type: "text"},
				{Key: "rejection_patterns", Label: "Rejection Patterns (comma-separated, case-insensitive)", Value: strings.Join(cfg.Crawler.RejectionPatterns, ", "), Type: "text"},
				{Key: "strip_query_params", Label: "Strip Query Params (comma-separated, case-insensitive)", Value: strings.Join(cfg.Crawler.StripQueryParams, ", "), Type: "text"},
				{Key: "pdf_heading_patterns", Label: "PDF Heading Patterns (comma-separated)", Value: strings.Join(cfg.Crawler.PDF.HeadingPatterns, ", "), Type: "text"},
			},
		},
		{
			Name: "object_store",
			Fields: []templates.ConfigField{
				{Key: "endpoint", Label: "Endpoint", Value: cfg.ObjectStore.Endpoint, Type: "text"},
				{Key: "bucket", Label: "Bucket", Value: cfg.ObjectStore.Bucket, Type: "text"},
				{Key: "access_key", Label: "Access Key", Value: cfg.ObjectStore.AccessKey, Type: "text"},
				{Key: "secret_key", Label: "Secret Key", Value: cfg.ObjectStore.SecretKey, Type: "text"},
				{Key: "region", Label: "Region", Value: cfg.ObjectStore.Region, Type: "text"},
			},
		},
	}
}

func formInt(r *http.Request, key string, fallback int) int {
	v := r.FormValue(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func formFloat(r *http.Request, key string, fallback float64) float64 {
	v := r.FormValue(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return f
}

func effectivePrompt(cfg *config.Config) string {
	if cfg.Agent.SystemPrompt != "" {
		return cfg.Agent.SystemPrompt
	}
	return prompts.Agent
}

func splitTrimCSV(s string) []string {
	parts := strings.Split(s, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
