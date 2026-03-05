package admin

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/config"
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

	switch section {
	case "server":
		h.config.Server.Host = r.FormValue("host")
		h.config.Server.Port = formInt(r, "port", h.config.Server.Port)
		h.config.Server.ReadTimeout = formInt(r, "read_timeout", h.config.Server.ReadTimeout)
		h.config.Server.WriteTimeout = formInt(r, "write_timeout", h.config.Server.WriteTimeout)
	case "rag":
		h.config.RAG.TopK = formInt(r, "top_k", h.config.RAG.TopK)
		h.config.RAG.SimilarityThreshold = formFloat(r, "similarity_threshold", h.config.RAG.SimilarityThreshold)
		h.config.RAG.MaxContextTokens = formInt(r, "max_context_tokens", h.config.RAG.MaxContextTokens)
	case "chunking":
		h.config.Chunking.ChunkSize = formInt(r, "chunk_size", h.config.Chunking.ChunkSize)
		h.config.Chunking.ChunkOverlap = formInt(r, "chunk_overlap", h.config.Chunking.ChunkOverlap)
	case "session":
		h.config.Session.MaxMessages = formInt(r, "max_messages", h.config.Session.MaxMessages)
		h.config.Session.TTLSeconds = formInt(r, "ttl_seconds", h.config.Session.TTLSeconds)
	case "embedding":
		h.config.Embedding.Backend = r.FormValue("backend")
		h.config.Embedding.Model = r.FormValue("model")
		h.config.Embedding.Dimension = formInt(r, "dimension", h.config.Embedding.Dimension)
	case "chat":
		h.config.Chat.Backend = r.FormValue("backend")
		h.config.Chat.Model = r.FormValue("model")
		h.config.Chat.MaxTokens = formInt(r, "max_tokens", h.config.Chat.MaxTokens)
		h.config.Chat.Temperature = formFloat(r, "temperature", h.config.Chat.Temperature)
	case "agent":
		h.config.Agent.Backend = r.FormValue("backend")
		h.config.Agent.Model = r.FormValue("model")
		h.config.Agent.MaxTokens = formInt(r, "max_tokens", h.config.Agent.MaxTokens)
		h.config.Agent.Temperature = formFloat(r, "temperature", h.config.Agent.Temperature)
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
}

func configToSections(cfg *config.Config) []templates.ConfigSection {
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
				{Key: "backend", Label: "Backend Name", Value: cfg.Embedding.Backend, Type: "text"},
				{Key: "model", Label: "Model", Value: cfg.Embedding.Model, Type: "text"},
				{Key: "dimension", Label: "Dimension", Value: strconv.Itoa(cfg.Embedding.Dimension), Type: "number"},
			},
		},
		{
			Name: "chat",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend Name", Value: cfg.Chat.Backend, Type: "text"},
				{Key: "model", Label: "Model", Value: cfg.Chat.Model, Type: "text"},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Chat.MaxTokens), Type: "number"},
				{Key: "temperature", Label: "Temperature", Value: fmt.Sprintf("%.2f", cfg.Chat.Temperature), Type: "text"},
			},
		},
		{
			Name: "agent",
			Fields: []templates.ConfigField{
				{Key: "backend", Label: "Backend Name", Value: cfg.Agent.Backend, Type: "text"},
				{Key: "model", Label: "Model", Value: cfg.Agent.Model, Type: "text"},
				{Key: "max_tokens", Label: "Max Tokens", Value: strconv.Itoa(cfg.Agent.MaxTokens), Type: "number"},
				{Key: "temperature", Label: "Temperature", Value: fmt.Sprintf("%.2f", cfg.Agent.Temperature), Type: "text"},
			},
		},
		{
			Name: "rag",
			Fields: []templates.ConfigField{
				{Key: "top_k", Label: "Top K", Value: strconv.Itoa(cfg.RAG.TopK), Type: "number"},
				{Key: "similarity_threshold", Label: "Similarity Threshold", Value: fmt.Sprintf("%.2f", cfg.RAG.SimilarityThreshold), Type: "text"},
				{Key: "max_context_tokens", Label: "Max Context Tokens", Value: strconv.Itoa(cfg.RAG.MaxContextTokens), Type: "number"},
			},
		},
		{
			Name: "chunking",
			Fields: []templates.ConfigField{
				{Key: "chunk_size", Label: "Chunk Size", Value: strconv.Itoa(cfg.Chunking.ChunkSize), Type: "number"},
				{Key: "chunk_overlap", Label: "Chunk Overlap", Value: strconv.Itoa(cfg.Chunking.ChunkOverlap), Type: "number"},
			},
		},
		{
			Name: "session",
			Fields: []templates.ConfigField{
				{Key: "max_messages", Label: "Max Messages", Value: strconv.Itoa(cfg.Session.MaxMessages), Type: "number"},
				{Key: "ttl_seconds", Label: "TTL (seconds)", Value: strconv.Itoa(cfg.Session.TTLSeconds), Type: "number"},
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
