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

// ConfigPage handles GET /admin/config and GET /admin/config/{section}. When
// no section is given, the first one in the layout becomes the default. The
// HTMX fragment path returns the whole shell (sidebar + content) so the
// active-link highlight updates on every navigation; the full-page path wraps
// the same shell in the layout.
func (h *Handler) ConfigPage(w http.ResponseWriter, r *http.Request) {
	sections := configToSections(h.config)
	if len(sections) == 0 {
		http.Error(w, "no config sections", http.StatusInternalServerError)
		return
	}

	name := chi.URLParam(r, "section")
	if name == "" {
		name = sections[0].Name
	}
	active := findSection(sections, name)
	if active == nil {
		http.NotFound(w, r)
		return
	}

	page := templates.ConfigPageTempl(sections, *active)
	fragment := templates.ConfigShell(sections, *active)
	render(w, r, page, fragment)
}

// ConfigSave handles POST /admin/config/{section}.
func (h *Handler) ConfigSave(w http.ResponseWriter, r *http.Request) {
	section := chi.URLParam(r, "section")
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	// Snapshot before mutation so a failed Validate() can revert in-memory state
	// without a deep copy. All slice fields (Backends, Crawler.Seeds, etc.) are
	// fully replaced rather than appended in place during the switch below, so
	// restoring the snapshot's slice headers leaves no dangling references.
	oldCfg := *h.config

	var embeddingInvalidated bool
	switch section {
	case "server":
		h.config.Server.Host = r.FormValue("host")
		h.config.Server.Port = formInt(r, "port", h.config.Server.Port)
		h.config.Server.ReadTimeout = formInt(r, "read_timeout", h.config.Server.ReadTimeout)
		h.config.Server.WriteTimeout = formInt(r, "write_timeout", h.config.Server.WriteTimeout)
		h.config.Server.IdleTimeout = formDurationSeconds(r, "idle_timeout_s", h.config.Server.IdleTimeout)
		h.config.Server.ShutdownTimeout = formDurationSeconds(r, "shutdown_timeout_s", h.config.Server.ShutdownTimeout)
	case "admin":
		h.config.Admin.BackendProbeTimeout = formDurationSeconds(r, "backend_probe_timeout_s", h.config.Admin.BackendProbeTimeout)
		h.config.Admin.BackendModelsTimeout = formDurationSeconds(r, "backend_models_timeout_s", h.config.Admin.BackendModelsTimeout)
		h.config.Admin.ChatStreamTimeout = formDurationSeconds(r, "chat_stream_timeout_s", h.config.Admin.ChatStreamTimeout)
	case "retrieval":
		h.config.Retrieval.TopK = formInt(r, "top_k", h.config.Retrieval.TopK)
		h.config.Retrieval.MaxContextTokens = formInt(r, "max_context_tokens", h.config.Retrieval.MaxContextTokens)
		if v := strings.TrimSpace(r.FormValue("retrieval_mode")); v != "" {
			h.config.Retrieval.RetrievalMode = v
		}
	case "reranker":
		h.config.Reranker.Enabled = r.FormValue("enabled") == "true"
		h.config.Reranker.Backend = r.FormValue("backend")
		h.config.Reranker.Model = r.FormValue("model")
		h.config.Reranker.TopN = formInt(r, "top_n", h.config.Reranker.TopN)
		h.config.Reranker.MaxTokens = formInt(r, "max_tokens", h.config.Reranker.MaxTokens)
		h.config.Reranker.ReservedTokens = formInt(r, "reserved_tokens", h.config.Reranker.ReservedTokens)
		h.config.Reranker.Timeout = formDurationSeconds(r, "timeout_s", h.config.Reranker.Timeout)
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
		oldModel := h.config.Embedding.Model
		h.config.Embedding.Backend = r.FormValue("backend")
		h.config.Embedding.Model = r.FormValue("model")
		h.config.Embedding.Dimension = formInt(r, "dimension", h.config.Embedding.Dimension)
		h.config.Embedding.MaxTokens = formInt(r, "max_tokens", h.config.Embedding.MaxTokens)
		h.config.Embedding.BatchSize = formInt(r, "batch_size", h.config.Embedding.BatchSize)
		// Either a dimension or a model change invalidates existing embeddings:
		// dimension breaks cosine math outright, model change keeps the math
		// valid but the vectors are now in a different semantic space, so
		// retrieval quality degrades silently. Skip the warning on first save
		// (oldDim/oldModel == zero value) so initial setup isn't noisy.
		dimensionChanged := oldDim != 0 && h.config.Embedding.Dimension != oldDim
		modelChanged := oldModel != "" && h.config.Embedding.Model != oldModel
		embeddingInvalidated = dimensionChanged || modelChanged
	case "agent":
		h.config.Agent.Backend = r.FormValue("backend")
		h.config.Agent.Model = r.FormValue("model")
		h.config.Agent.MaxTokens = formInt(r, "max_tokens", h.config.Agent.MaxTokens)
		h.config.Agent.Temperature = formFloat(r, "temperature", h.config.Agent.Temperature)
		h.config.Agent.ContextWindow = formInt(r, "context_window", h.config.Agent.ContextWindow)
		h.config.Agent.MaxIterations = formInt(r, "max_iterations", h.config.Agent.MaxIterations)
		h.config.Agent.ToolOutputMaxTokens = formInt(r, "tool_output_max_tokens", h.config.Agent.ToolOutputMaxTokens)
		h.config.Agent.Passthrough = r.FormValue("passthrough") == "true"
		submitted := strings.TrimSpace(r.FormValue("system_prompt"))
		if submitted == "" || submitted == strings.TrimSpace(prompts.Agent) {
			h.config.Agent.SystemPrompt = ""
		} else {
			h.config.Agent.SystemPrompt = submitted
		}
	case "agent.vision":
		h.config.Agent.Vision.MaxImagesPerCall = formInt(r, "max_images_per_call", h.config.Agent.Vision.MaxImagesPerCall)
		h.config.Agent.Vision.MaxImageBytesPerCall = int64(formInt(r, "max_image_bytes_per_call", int(h.config.Agent.Vision.MaxImageBytesPerCall)))
		h.config.Agent.Vision.PublicBaseURL = strings.TrimSpace(r.FormValue("public_base_url"))
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
		h.config.Crawler.PDF.RenderDPI = formInt(r, "pdf_render_dpi", h.config.Crawler.PDF.RenderDPI)
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

	// Cross-section invariants (e.g. chunk_size <= max_context_tokens). Render
	// the rejected values so the user can see what they typed, then revert the
	// in-memory config so the live pipeline keeps the previous valid state.
	if err := h.config.Validate(); err != nil {
		rejected := findSection(configToSections(h.config), section)
		*h.config = oldCfg
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if rejected != nil {
			templates.ConfigSectionForm(*rejected).Render(r.Context(), w)
		}
		templates.Toast("Invalid config: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	if err := config.WriteTo(h.config, h.configPath); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if saved := findSection(configToSections(h.config), section); saved != nil {
			templates.ConfigSectionForm(*saved).Render(r.Context(), w)
		}
		templates.Toast("Error saving config: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	h.fireConfigSave()
	h.logger.Info().Str("section", section).Msg("config saved")

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if saved := findSection(configToSections(h.config), section); saved != nil {
		templates.ConfigSectionForm(*saved).Render(r.Context(), w)
	}
	templates.Toast(fmt.Sprintf("Section '%s' saved", section), "success").Render(r.Context(), w)
	if embeddingInvalidated {
		templates.Toast("Embedding model or dimension changed — reprocess crawled pages to refresh stored embeddings", "warning").Render(r.Context(), w)
	}
}

// configToSections groups config fields into UI cards. Each section carries a
// Group label so the page can render Infrastructure / Pipeline / Ingest as
// separate visual blocks.
func configToSections(cfg *config.Config) []templates.ConfigSection {
	backendNames := make([]string, len(cfg.Backends))
	for i, b := range cfg.Backends {
		backendNames[i] = b.Name
	}

	return []templates.ConfigSection{
		// ---------- Infrastructure ----------
		{
			Name:  "server",
			Title: "Server",
			Group: "Infrastructure",
			Fields: []templates.ConfigField{
				{Key: "host", Label: "Host", Value: cfg.Server.Host, Type: "text"},
				{Key: "port", Label: "Port", Value: strconv.Itoa(cfg.Server.Port), Type: "number"},
				{Key: "read_timeout", Label: "Read Timeout (s)", Value: strconv.Itoa(cfg.Server.ReadTimeout), Type: "number"},
				{Key: "write_timeout", Label: "Write Timeout (s)", Value: strconv.Itoa(cfg.Server.WriteTimeout), Type: "number"},
				{Key: "idle_timeout_s", Label: "Idle Timeout (s)", Value: strconv.FormatInt(int64(cfg.Server.IdleTimeout/time.Second), 10), Type: "number", Hint: "HTTP keep-alive idle timeout"},
				{Key: "shutdown_timeout_s", Label: "Shutdown Timeout (s)", Value: strconv.FormatInt(int64(cfg.Server.ShutdownTimeout/time.Second), 10), Type: "number", Hint: "Grace period for in-flight requests on shutdown"},
			},
		},
		{
			Name:  "admin",
			Title: "Admin",
			Group: "Infrastructure",
			Fields: []templates.ConfigField{
				{Key: "backend_probe_timeout_s", Label: "Backend Probe Timeout (s)", Value: strconv.FormatInt(int64(cfg.Admin.BackendProbeTimeout/time.Second), 10), Type: "number", Hint: "Per-request timeout for backend health checks"},
				{Key: "backend_models_timeout_s", Label: "Backend Models Timeout (s)", Value: strconv.FormatInt(int64(cfg.Admin.BackendModelsTimeout/time.Second), 10), Type: "number", Hint: "Per-request timeout for /models listing"},
				{Key: "chat_stream_timeout_s", Label: "Chat Stream Timeout (s)", Value: strconv.FormatInt(int64(cfg.Admin.ChatStreamTimeout/time.Second), 10), Type: "number", Hint: "Max lifetime of a chat-stream goroutine (orphaned-goroutine safety net)"},
			},
		},
		{
			Name:  "database",
			Title: "Database",
			Group: "Infrastructure",
			Fields: []templates.ConfigField{
				{Key: "path", Label: "Path", Value: cfg.Database.Path, Type: "text"},
			},
		},
		{
			Name:  "object_store",
			Title: "Object Store",
			Group: "Infrastructure",
			Fields: []templates.ConfigField{
				{Key: "endpoint", Label: "Endpoint", Value: cfg.ObjectStore.Endpoint, Type: "text"},
				{Key: "bucket", Label: "Bucket", Value: cfg.ObjectStore.Bucket, Type: "text"},
				{Key: "access_key", Label: "Access Key", Value: cfg.ObjectStore.AccessKey, Type: "text"},
				{Key: "secret_key", Label: "Secret Key", Value: cfg.ObjectStore.SecretKey, Type: "text"},
				{Key: "region", Label: "Region", Value: cfg.ObjectStore.Region, Type: "text"},
			},
		},

		// ---------- Pipeline ----------
		{
			Name:  "chunking",
			Title: "Chunking",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "max_chunk_size", Label: "Max Chunk Size (tokens)", Value: strconv.Itoa(cfg.Chunking.MaxChunkSize), Type: "number"},
				{Key: "chunk_overlap", Label: "Chunk Overlap (tokens)", Value: strconv.Itoa(cfg.Chunking.ChunkOverlap), Type: "number"},
				{Key: "min_chunk_size", Label: "Min Chunk Size (tokens)", Value: strconv.Itoa(cfg.Chunking.MinChunkSize), Type: "number"},
				{Key: "separators", Label: "Separators (comma-separated)", Value: strings.Join(cfg.Chunking.Separators, ","), Type: "text"},
			},
		},
		{
			Name:  "embedding",
			Title: "Embedding",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Embedding.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Embedding.Model, Type: "model", Options: backendNames},
				{Key: "dimension", Label: "Dimension", Value: strconv.Itoa(cfg.Embedding.Dimension), Type: "number"},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Embedding.MaxTokens), Type: "number"},
				{Key: "batch_size", Label: "Batch Size", Value: strconv.Itoa(cfg.Embedding.BatchSize), Type: "number"},
			},
		},
		{
			Name:  "retrieval",
			Title: "Retrieval",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "top_k", Label: "Top K", Value: strconv.Itoa(cfg.Retrieval.TopK), Type: "number"},
				{Key: "max_context_tokens", Label: "Max Context Tokens", Value: strconv.Itoa(cfg.Retrieval.MaxContextTokens), Type: "number"},
				{
					Key:     "retrieval_mode",
					Label:   "Retrieval Mode",
					Value:   cfg.Retrieval.RetrievalMode,
					Type:    "enum",
					Options: []string{"dense", "hybrid"},
					Hint:    "dense = cosine over embeddings; hybrid = dense + BM25 via reciprocal-rank fusion (ADR-008)",
				},
			},
		},
		{
			Name:  "reranker",
			Title: "Reranker",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{
					Key:   "enabled",
					Label: "Enabled",
					Value: strconv.FormatBool(cfg.Reranker.Enabled),
					Type:  "checkbox",
					Hint:  "When enabled, re-scores retrieval candidates with a cross-encoder. Backend MUST be llama.cpp (the /rerank endpoint is a llama.cpp/Jina extension).",
				},
				{Key: "backend", Label: "Backend", Value: cfg.Reranker.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Reranker.Model, Type: "model", Options: backendNames},
				{Key: "top_n", Label: "Candidate Pool Size (top_n)", Value: strconv.Itoa(cfg.Reranker.TopN), Type: "number", Hint: "Must be >= retrieval.top_k. Typical: 30."},
				{Key: "max_tokens", Label: "Max Tokens (0 = auto-detect via /props)", Value: strconv.Itoa(cfg.Reranker.MaxTokens), Type: "number"},
				{Key: "reserved_tokens", Label: "Reserved Tokens", Value: strconv.Itoa(cfg.Reranker.ReservedTokens), Type: "number", Hint: "BOS/EOS/separator budget held back from per-doc truncation."},
				{Key: "timeout_s", Label: "Timeout (s)", Value: strconv.FormatInt(int64(cfg.Reranker.Timeout/time.Second), 10), Type: "number", Hint: "Per-request deadline. Default 30s absorbs cold-load on the router."},
			},
		},
		{
			Name:  "agent",
			Title: "Agent",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Agent.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Agent.Model, Type: "model", Options: backendNames},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Agent.MaxTokens), Type: "number"},
				{Key: "temperature", Label: "Temperature", Value: fmt.Sprintf("%.2f", cfg.Agent.Temperature), Type: "text"},
				{Key: "context_window", Label: "Context Window (auto-detected)", Value: strconv.Itoa(cfg.Agent.ContextWindow), Type: "number"},
				{Key: "max_iterations", Label: "Max Tool Iterations", Value: strconv.Itoa(cfg.Agent.MaxIterations), Type: "number"},
				{Key: "tool_output_max_tokens", Label: "Tool Output Max Tokens (0 = auto, context_window/4)", Value: strconv.Itoa(cfg.Agent.ToolOutputMaxTokens), Type: "number"},
				{
					Key:   "passthrough",
					Label: "Passthrough mode",
					Value: strconv.FormatBool(cfg.Agent.Passthrough),
					Type:  "checkbox",
					Hint:  "Accept client-supplied message lists directly without session management.",
				},
				{Key: "system_prompt", Label: "System Prompt (clear to restore default)", Value: effectivePrompt(cfg), Type: "textarea", Hint: "Template variables: {{.DocumentCount}}, {{.ChunkCount}}"},
			},
		},
		{
			Name:  "agent.vision",
			Title: "Vision",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "max_images_per_call", Label: "Max Images Per Tool Call", Value: strconv.Itoa(cfg.Agent.Vision.MaxImagesPerCall), Type: "number"},
				{Key: "max_image_bytes_per_call", Label: "Max Image-URL Bytes Per Call", Value: strconv.FormatInt(cfg.Agent.Vision.MaxImageBytesPerCall, 10), Type: "number"},
				{Key: "public_base_url", Label: "Public Base URL", Value: cfg.Agent.Vision.PublicBaseURL, Type: "text", Hint: "Absolute URL the LLM uses to fetch /v1/artifacts/... e.g. http://localhost:8080. Required for image attachment."},
			},
		},
		{
			Name:  "session",
			Title: "Session",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "max_messages", Label: "Max Messages", Value: strconv.Itoa(cfg.Session.MaxMessages), Type: "number"},
				{Key: "ttl_seconds", Label: "TTL (seconds)", Value: strconv.Itoa(cfg.Session.TTLSeconds), Type: "number"},
			},
		},

		// ---------- Ingest ----------
		{
			Name:  "crawler",
			Title: "Crawler",
			Group: "Ingest",
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
				{Key: "pdf_render_dpi", Label: "PDF Render DPI (pdftoppm; 0 disables image rendering)", Value: strconv.Itoa(cfg.Crawler.PDF.RenderDPI), Type: "number"},
			},
		},
	}
}

// findSection looks up a section in a slice produced by configToSections.
// Returns nil if no section has the given name.
func findSection(sections []templates.ConfigSection, name string) *templates.ConfigSection {
	for i := range sections {
		if sections[i].Name == name {
			return &sections[i]
		}
	}
	return nil
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

// formDurationSeconds reads an integer-seconds form field and returns a
// time.Duration. Empty or invalid input falls back to the existing value.
func formDurationSeconds(r *http.Request, key string, fallback time.Duration) time.Duration {
	v := r.FormValue(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return time.Duration(n) * time.Second
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
