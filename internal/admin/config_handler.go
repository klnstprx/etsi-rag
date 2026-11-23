package admin

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/a-h/templ"
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
	sections := configToSections(h.config, h.capacities)
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
		h.config.Server.RequestTimeout = formInt(r, "request_timeout", h.config.Server.RequestTimeout)
		h.config.Server.IdleTimeout = formDuration(r, "idle_timeout_s", time.Second, h.config.Server.IdleTimeout)
		h.config.Server.ShutdownTimeout = formDuration(r, "shutdown_timeout_s", time.Second, h.config.Server.ShutdownTimeout)
	case "admin":
		h.config.Admin.BackendProbeTimeout = formDuration(r, "backend_probe_timeout_s", time.Second, h.config.Admin.BackendProbeTimeout)
		h.config.Admin.BackendModelsTimeout = formDuration(r, "backend_models_timeout_s", time.Second, h.config.Admin.BackendModelsTimeout)
		h.config.Admin.ChatStreamTimeout = formDuration(r, "chat_stream_timeout_s", time.Second, h.config.Admin.ChatStreamTimeout)
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
		h.config.Reranker.Timeout = formDuration(r, "timeout_s", time.Second, h.config.Reranker.Timeout)
		h.config.Reranker.TitleInText = r.FormValue("title_in_text") == "true"
	case "chunking":
		h.config.Chunking.MaxChunkSize = formInt(r, "max_chunk_size", h.config.Chunking.MaxChunkSize)
		h.config.Chunking.ChunkOverlap = formInt(r, "chunk_overlap", h.config.Chunking.ChunkOverlap)
		h.config.Chunking.MinChunkSize = formInt(r, "min_chunk_size", h.config.Chunking.MinChunkSize)
		if v := r.FormValue("separators"); v != "" {
			h.config.Chunking.Separators = strings.Split(v, ",")
		}
		h.config.Chunking.TitleInSearchText = r.FormValue("title_in_search_text") == "true"
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
		h.config.Agent.HistoryReserveTokens = formInt(r, "history_reserve_tokens", h.config.Agent.HistoryReserveTokens)
		h.config.Agent.UserReserveTokens = formInt(r, "user_reserve_tokens", h.config.Agent.UserReserveTokens)
		submitted := strings.TrimSpace(r.FormValue("system_prompt"))
		if submitted == "" || submitted == strings.TrimSpace(prompts.Agent) {
			h.config.Agent.SystemPrompt = ""
		} else {
			h.config.Agent.SystemPrompt = submitted
		}
	case "agent.vision":
		h.config.Agent.Vision.Enabled = r.FormValue("enabled") == "true"
		h.config.Agent.Vision.MaxImagesPerCall = formInt(r, "max_images_per_call", h.config.Agent.Vision.MaxImagesPerCall)
		h.config.Agent.Vision.MaxImageBytesPerCall = int64(formInt(r, "max_image_bytes_per_call", int(h.config.Agent.Vision.MaxImageBytesPerCall)))
		h.config.Agent.Vision.ImageTransport = strings.TrimSpace(r.FormValue("image_transport"))
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
		h.config.Crawler.RequestDelay = formDuration(r, "request_delay_ms", time.Millisecond, h.config.Crawler.RequestDelay)
		h.config.Crawler.RequestTimeout = formDuration(r, "request_timeout_s", time.Second, h.config.Crawler.RequestTimeout)
		h.config.Crawler.ThrottledPaths = formCSV(r, "throttled_paths")
		h.config.Crawler.ThrottledPathDelay = formDuration(r, "throttled_path_delay_s", time.Second, h.config.Crawler.ThrottledPathDelay)
		h.config.Crawler.RateLimitRetries = formInt(r, "rate_limit_retries", h.config.Crawler.RateLimitRetries)
		h.config.Crawler.RateLimitBackoff = formDuration(r, "rate_limit_backoff_s", time.Second, h.config.Crawler.RateLimitBackoff)
		h.config.Crawler.RateLimitMaxWait = formDuration(r, "rate_limit_max_wait_s", time.Second, h.config.Crawler.RateLimitMaxWait)
		h.config.Crawler.JSPatterns = formCSV(r, "js_patterns")
		h.config.Crawler.PDF.HeadingPatterns = formCSV(r, "pdf_heading_patterns")
		h.config.Crawler.PDF.RenderDPI = formInt(r, "pdf_render_dpi", h.config.Crawler.PDF.RenderDPI)
		h.config.Crawler.RejectionPatterns = formCSV(r, "rejection_patterns")
		h.config.Crawler.StripQueryParams = formCSV(r, "strip_query_params")
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
		rejected := *h.config
		*h.config = oldCfg
		h.writeSection(w, r, &rejected, section, templates.Toast("Invalid config: "+err.Error(), "error"))
		return
	}

	if err := h.saveConfig(); err != nil {
		h.logger.Error().Err(err).Msg("writing config")
		h.writeSection(w, r, h.config, section, templates.Toast("Error saving config: "+err.Error(), "error"))
		return
	}

	h.logger.Info().Str("section", section).Msg("config saved")

	toasts := []templ.Component{templates.Toast(fmt.Sprintf("Section '%s' saved", section), "success")}
	if embeddingInvalidated {
		toasts = append(toasts, templates.Toast("Embedding model or dimension changed — reprocess crawled pages to refresh stored embeddings", "warning"))
	}
	h.writeSection(w, r, h.config, section, toasts...)
}

// writeSection renders the named section's form from cfg (when it exists)
// followed by toasts, the reply to every config save attempt.
func (h *Handler) writeSection(w http.ResponseWriter, r *http.Request, cfg *config.Config, name string, toasts ...templ.Component) {
	var parts []templ.Component
	if s := findSection(configToSections(cfg, h.capacities), name); s != nil {
		parts = append(parts, templates.ConfigSectionForm(*s))
	}
	writeHTML(w, r, append(parts, toasts...)...)
}

// configToSections groups config fields into UI cards. Each section carries a
// Group label so the page can render Infrastructure / Pipeline / Ingest as
// separate visual blocks.
func configToSections(cfg *config.Config, caps Capacities) []templates.ConfigSection {
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
				{Key: "write_timeout", Label: "Write Timeout (s)", Value: strconv.Itoa(cfg.Server.WriteTimeout), Type: "number", Hint: "Outer http.Server write deadline; should be larger than request_timeout"},
				{Key: "request_timeout", Label: "Request Timeout (s)", Value: strconv.Itoa(cfg.Server.RequestTimeout), Type: "number", Hint: "Per-request handler ceiling; bounds the agent tool-calling loop on /v1/chat/completions"},
				{Key: "idle_timeout_s", Label: "Idle Timeout (s)", Value: formatDuration(cfg.Server.IdleTimeout, time.Second), Type: "number", Hint: "HTTP keep-alive idle timeout"},
				{Key: "shutdown_timeout_s", Label: "Shutdown Timeout (s)", Value: formatDuration(cfg.Server.ShutdownTimeout, time.Second), Type: "number", Hint: "Grace period for in-flight requests on shutdown"},
			},
		},
		{
			Name:  "admin",
			Title: "Admin",
			Group: "Infrastructure",
			Fields: []templates.ConfigField{
				{Key: "backend_probe_timeout_s", Label: "Backend Probe Timeout (s)", Value: formatDuration(cfg.Admin.BackendProbeTimeout, time.Second), Type: "number", Hint: "Per-request timeout for backend health checks"},
				{Key: "backend_models_timeout_s", Label: "Backend Models Timeout (s)", Value: formatDuration(cfg.Admin.BackendModelsTimeout, time.Second), Type: "number", Hint: "Per-request timeout for /models listing"},
				{Key: "chat_stream_timeout_s", Label: "Chat Stream Timeout (s)", Value: formatDuration(cfg.Admin.ChatStreamTimeout, time.Second), Type: "number", Hint: "Max lifetime of a chat-stream goroutine (orphaned-goroutine safety net)"},
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
				{Key: "title_in_search_text", Label: "Title in search text", Value: strconv.FormatBool(cfg.Chunking.TitleInSearchText), Type: "checkbox", Hint: "Embed and keyword-index each chunk with its document title ahead of the heading breadcrumb. Reprocess after changing it."},
			},
		},
		{
			Name:  "embedding",
			Title: "Embedding",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Embedding.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Embedding.Model, Type: "model"},
				{Key: "dimension", Label: "Dimension", Value: strconv.Itoa(cfg.Embedding.Dimension), Type: "number"},
				{Key: "max_tokens", Label: "Max Tokens (0 = auto-detect)", Value: strconv.Itoa(cfg.Embedding.MaxTokens), Type: "number", Hint: effectiveNote("0 follows the llama.cpp /props probe; a non-zero value overrides it. A 512-token floor applies when neither is available.", caps, Capacities.EmbeddingMaxTokens)},
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
					Hint:    "dense = cosine over embeddings; hybrid = dense + BM25 via reciprocal-rank fusion",
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
				{Key: "model", Label: "Model", Value: cfg.Reranker.Model, Type: "model"},
				{Key: "top_n", Label: "Candidate Pool Size (top_n)", Value: strconv.Itoa(cfg.Reranker.TopN), Type: "number", Hint: "Must be >= retrieval.top_k. Typical: 30."},
				{Key: "max_tokens", Label: "Max Tokens (0 = auto-detect)", Value: strconv.Itoa(cfg.Reranker.MaxTokens), Type: "number", Hint: effectiveNote("Per-doc truncation budget. 0 follows the llama.cpp /props probe; a non-zero value overrides it.", caps, Capacities.RerankerMaxTokens)},
				{Key: "reserved_tokens", Label: "Reserved Tokens", Value: strconv.Itoa(cfg.Reranker.ReservedTokens), Type: "number", Hint: "BOS/EOS/separator budget held back from per-doc truncation."},
				{Key: "timeout_s", Label: "Timeout (s)", Value: formatDuration(cfg.Reranker.Timeout, time.Second), Type: "number", Hint: "Per-request deadline. Default 30s absorbs cold-load on the router."},
				{Key: "title_in_text", Label: "Title in rerank text", Value: strconv.FormatBool(cfg.Reranker.TitleInText), Type: "checkbox", Hint: "Send each candidate with its document title and heading breadcrumb ahead of the body, as the search index sees it."},
			},
		},
		{
			Name:  "agent",
			Title: "Agent",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend", Value: cfg.Agent.Backend, Type: "select", Options: backendNames},
				{Key: "model", Label: "Model", Value: cfg.Agent.Model, Type: "model"},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Agent.MaxTokens), Type: "number"},
				{Key: "temperature", Label: "Temperature", Value: fmt.Sprintf("%.2f", cfg.Agent.Temperature), Type: "text"},
				{Key: "context_window", Label: "Context Window (0 = auto-detect)", Value: strconv.Itoa(cfg.Agent.ContextWindow), Type: "number", Hint: effectiveNote("0 follows the llama.cpp /props probe automatically (re-read at startup and on every save); a non-zero value overrides it.", caps, Capacities.AgentContextWindow)},
				{Key: "max_iterations", Label: "Max Tool Iterations", Value: strconv.Itoa(cfg.Agent.MaxIterations), Type: "number"},
				{Key: "tool_output_max_tokens", Label: "Tool Output Max Tokens (0 = auto from budget)", Value: strconv.Itoa(cfg.Agent.ToolOutputMaxTokens), Type: "number", Hint: "Caps tokens per tool result. 0 derives the cap from the token budget pool (context window minus generation, fixed prompt/schema cost, and the history/user reserves)."},
				{Key: "history_reserve_tokens", Label: "History Reserve (tokens)", Value: strconv.Itoa(cfg.Agent.HistoryReserveTokens), Type: "number", Hint: "Headroom held back from the context window for session history when deriving the tool budget."},
				{Key: "user_reserve_tokens", Label: "User Reserve (tokens)", Value: strconv.Itoa(cfg.Agent.UserReserveTokens), Type: "number", Hint: "Headroom held back from the context window for the incoming user message when deriving the tool budget."},
				{Key: "system_prompt", Label: "System Prompt (clear to restore default)", Value: effectivePrompt(cfg), Type: "textarea", Hint: "Template variables: {{.DocumentCount}}, {{.ChunkCount}}"},
			},
		},
		{
			Name:  "agent.vision",
			Title: "Vision",
			Group: "Pipeline",
			Fields: []templates.ConfigField{
				{Key: "enabled", Label: "Vision enabled", Value: strconv.FormatBool(cfg.Agent.Vision.Enabled), Type: "checkbox", Hint: "Tick when the configured agent.model accepts image_url content blocks (mmproj for llama.cpp; -vision/-mm models on hosted providers)."},
				{Key: "max_images_per_call", Label: "Max Images Per Tool Call", Value: strconv.Itoa(cfg.Agent.Vision.MaxImagesPerCall), Type: "number"},
				{Key: "max_image_bytes_per_call", Label: "Max Image-URL Bytes Per Call", Value: strconv.FormatInt(cfg.Agent.Vision.MaxImageBytesPerCall, 10), Type: "number"},
				{Key: "image_transport", Label: "Image Transport", Value: cfg.Agent.Vision.ImageTransport, Type: "select", Options: []string{"inline", "url"}, Hint: "inline sends the image bytes in the request (no fetch by the backend); url sends Public Base URL links the backend fetches."},
				{Key: "public_base_url", Label: "Public Base URL", Value: cfg.Agent.Vision.PublicBaseURL, Type: "text", Hint: "url transport only: absolute URL the LLM uses to fetch /v1/artifacts/... e.g. http://localhost:8080. llama.cpp also needs --media-allow-private-urls for a local address."},
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
				{Key: "request_delay_ms", Label: "Request Delay (ms)", Value: formatDuration(cfg.Crawler.RequestDelay, time.Millisecond), Type: "number"},
				{Key: "request_timeout_s", Label: "Request Timeout (s)", Value: formatDuration(cfg.Crawler.RequestTimeout, time.Second), Type: "number"},
				{Key: "throttled_paths", Label: "Throttled Paths (comma-separated)", Value: strings.Join(cfg.Crawler.ThrottledPaths, ", "), Type: "text", Hint: "Path substrings paced in their own lane per host, so a 429 on them does not hold the host's pages"},
				{Key: "throttled_path_delay_s", Label: "Throttled Path Delay (s)", Value: formatDuration(cfg.Crawler.ThrottledPathDelay, time.Second), Type: "number", Hint: "Minimum gap between requests to throttled paths of one host"},
				{Key: "rate_limit_retries", Label: "Rate Limit Retries", Value: strconv.Itoa(cfg.Crawler.RateLimitRetries), Type: "number", Hint: "Retries of a page answered 429 or 503 before it is recorded as failed"},
				{Key: "rate_limit_backoff_s", Label: "Rate Limit Backoff (s)", Value: formatDuration(cfg.Crawler.RateLimitBackoff, time.Second), Type: "number", Hint: "First wait when the server sends no Retry-After; doubles on each retry"},
				{Key: "rate_limit_max_wait_s", Label: "Rate Limit Max Wait (s)", Value: formatDuration(cfg.Crawler.RateLimitMaxWait, time.Second), Type: "number", Hint: "Cap on any single wait, Retry-After included"},
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

// effectiveNote appends the live probe-resolved value to a hint when a
// Capacities source is wired and reports a usable value.
func effectiveNote(hint string, caps Capacities, get func(Capacities) int) string {
	if caps == nil {
		return hint
	}
	if v := get(caps); v > 0 {
		return fmt.Sprintf("%s Effective now: %d tokens.", hint, v)
	}
	return hint
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

// formInt, formFloat and formDuration keep fallback when the field is empty
// or unparseable, so a partial form never zeroes a setting.
func formInt(r *http.Request, key string, fallback int) int {
	n, err := strconv.Atoi(r.FormValue(key))
	if err != nil {
		return fallback
	}
	return n
}

func formFloat(r *http.Request, key string, fallback float64) float64 {
	f, err := strconv.ParseFloat(r.FormValue(key), 64)
	if err != nil {
		return fallback
	}
	return f
}

// formDuration reads an integer field counted in unit (e.g. "timeout_s" in
// time.Second).
func formDuration(r *http.Request, key string, unit, fallback time.Duration) time.Duration {
	n, err := strconv.Atoi(r.FormValue(key))
	if err != nil {
		return fallback
	}
	return time.Duration(n) * unit
}

// formatDuration is formDuration's inverse for rendering the field value.
func formatDuration(d, unit time.Duration) string {
	return strconv.FormatInt(int64(d/unit), 10)
}

func effectivePrompt(cfg *config.Config) string {
	if cfg.Agent.SystemPrompt != "" {
		return cfg.Agent.SystemPrompt
	}
	return prompts.Agent
}

// formCSV splits a comma-separated field into trimmed entries; an empty
// field clears the list.
func formCSV(r *http.Request, key string) []string {
	v := r.FormValue(key)
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
