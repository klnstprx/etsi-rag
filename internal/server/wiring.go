package server

import (
	"context"
	"fmt"
	"strings"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/backend"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/tokencount"
)

// embedClientConfig pulls the [embedding] stage cfg out of the live config so
// callers don't restate the field-by-field literal at every call site
// (server.New, rebuildClients, recheck closures all need the same shape).
func (s *Server) embedClientConfig() (string, llm.EmbedClientConfig) {
	return s.cfg.Embedding.Backend, llm.EmbedClientConfig{
		Model:     s.cfg.Embedding.Model,
		BatchSize: s.cfg.Embedding.BatchSize,
	}
}

// chatClientConfig is the agent-stage equivalent of embedClientConfig.
func (s *Server) chatClientConfig() (string, llm.ChatClientConfig) {
	return s.cfg.Agent.Backend, llm.ChatClientConfig{
		Model:       s.cfg.Agent.Model,
		MaxTokens:   int64(s.cfg.Agent.MaxTokens),
		Temperature: s.cfg.Agent.Temperature,
	}
}

// resolveEmbedClient resolves a backend by name and builds an embedding client.
func (s *Server) resolveEmbedClient(backendName string, partialCfg llm.EmbedClientConfig) (*llm.EmbedClient, error) {
	b := s.cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for embedding", backendName)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	s.logger.Info().Str("stage", "embedding").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewEmbedClient(partialCfg), nil
}

// resolveChatClient resolves a backend by name and builds a chat client.
func (s *Server) resolveChatClient(backendName string, partialCfg llm.ChatClientConfig) (*llm.ChatClient, error) {
	b := s.cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for chat", backendName)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	s.logger.Info().Str("stage", "agent").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewChatClient(partialCfg), nil
}

// resolveRerankClient resolves a backend by name and builds a reranker client.
// The reranker is llama.cpp-only because /rerank is a llama.cpp/Jina extension
// not part of the OpenAI surface — config.Validate enforces this at load time
// but we re-check here so a TOML edited on disk can't bypass the safeguard.
func (s *Server) resolveRerankClient(backendName string, partialCfg llm.RerankClientConfig) (*llm.RerankClient, error) {
	b := s.cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for reranker", backendName)
	}
	if !b.IsLlamaCpp {
		return nil, fmt.Errorf("reranker backend %q must be is_llama_cpp=true (/rerank is not an OpenAI endpoint)", b.Name)
	}

	partialCfg.BaseURL = b.BaseURL
	partialCfg.APIKey = b.APIKey
	s.logger.Info().Str("stage", "reranker").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewRerankClient(partialCfg), nil
}

// attachReranker (re)builds the reranker client from the live config and
// installs it on the pipeline. Called both at startup and after every admin
// config save so toggling [reranker] enabled takes effect without a restart.
//
// Failure modes (disabled, missing backend, non-llama-cpp backend, missing
// counter) all degrade gracefully: the pipeline keeps the un-reranked path
// and the operator sees a warning in the logs.
func (s *Server) attachReranker() {
	if !s.cfg.Reranker.Enabled {
		s.pipeline.SetReranker(nil, 0)
		return
	}

	// Token counter is required for the context-budget guard; without it,
	// oversized docs would hit the cross-encoder's context window and the
	// /rerank call would fail at request time. Bail out if we can't resolve
	// one — failure here is a config issue worth surfacing.
	counter, err := s.resolveCounter(s.cfg.Reranker.Backend, s.cfg.Reranker.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("reranker disabled: no token counter")
		s.pipeline.SetReranker(nil, 0)
		return
	}

	rc, err := s.resolveRerankClient(s.cfg.Reranker.Backend, llm.RerankClientConfig{
		Model:          s.cfg.Reranker.Model,
		Counter:        counter,
		MaxTokens:      s.cfg.Reranker.MaxTokens,
		ReservedTokens: s.cfg.Reranker.ReservedTokens,
		Timeout:        s.cfg.Reranker.Timeout,
	})
	if err != nil {
		s.logger.Warn().Err(err).Msg("reranker disabled: client resolution failed")
		s.pipeline.SetReranker(nil, 0)
		return
	}

	s.pipeline.SetReranker(rc, s.cfg.Reranker.TopN)
	s.logger.Info().
		Str("backend", s.cfg.Reranker.Backend).
		Str("model", s.cfg.Reranker.Model).
		Int("top_n", s.cfg.Reranker.TopN).
		Int("max_tokens", s.cfg.Reranker.MaxTokens).
		Msg("reranker enabled")
}

// resolveCounter creates an appropriate token counter for the given backend.
func (s *Server) resolveCounter(backendName, model string) (tokencount.Counter, error) {
	b := s.cfg.GetBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for token counter", backendName)
	}
	if b.IsLlamaCpp {
		base := strings.TrimSuffix(b.BaseURL, "/v1")
		s.logger.Info().Str("backend", b.Name).Msg("using llama.cpp token counter")
		return tokencount.NewLlamaCppCounter(base, model, nil), nil
	}
	tc, err := tokencount.NewTiktokenCounter(model)
	if err != nil {
		return nil, fmt.Errorf("tiktoken init for model %q: %w", model, err)
	}
	s.logger.Info().Str("backend", b.Name).Str("model", model).Msg("using tiktoken counter")
	return tc, nil
}

// probeLlamaCppBackends probes each unique llama.cpp backend once and applies
// the results to the relevant stage configs.
func (s *Server) probeLlamaCppBackends() {
	type stageRef struct {
		name    string
		backend string
		model   string
	}
	stages := []stageRef{
		{"embedding", s.cfg.Embedding.Backend, s.cfg.Embedding.Model},
		{"agent", s.cfg.Agent.Backend, s.cfg.Agent.Model},
		{"reranker", s.cfg.Reranker.Backend, s.cfg.Reranker.Model},
	}

	probed := make(map[string]*backend.LlamaCppProps) // key: "backend:model"
	for _, st := range stages {
		if st.backend == "" {
			continue
		}
		key := st.backend + ":" + st.model
		if _, done := probed[key]; done {
			continue
		}
		b := s.cfg.GetBackend(st.backend)
		if b == nil || !b.IsLlamaCpp {
			continue
		}
		props, err := backend.FetchLlamaCppProps(context.Background(), b.BaseURL, st.model, nil)
		if err != nil {
			s.logger.Warn().Err(err).Str("backend", b.Name).Str("model", st.model).
				Msg("could not probe llama.cpp backend, using config values")
			continue
		}
		probed[key] = props
		s.logger.Info().Str("backend", b.Name).Str("model", st.model).
			Int("n_ctx", props.ContextSize).
			Int("n_embd", props.EmbeddingDim).
			Str("model_path", props.ModelPath).
			Str("model_alias", props.ModelAlias).
			Bool("supports_vision_detected", props.SupportsVision).
			Msg("probed llama.cpp backend")

		// Vision hint: never override the manual agent.vision.enabled flag,
		// but log a suggestion when the probe sees vision on the agent's
		// model and the flag is off, so the operator can flip it in admin
		// without re-reading the discovery output. Only meaningful for the
		// agent stage — embedding/reranker probes also pass through here.
		if props.SupportsVision && st.name == "agent" && !s.cfg.Agent.Vision.Enabled {
			s.logger.Info().Str("backend", b.Name).Str("model", st.model).
				Msg("hint: probe detected modalities.vision — consider enabling agent.vision.enabled in admin")
		}
	}

	// Apply embedding-specific results
	embKey := s.cfg.Embedding.Backend + ":" + s.cfg.Embedding.Model
	if props, ok := probed[embKey]; ok {
		if props.ContextSize > 0 {
			s.cfg.Embedding.MaxTokens = props.ContextSize
		}
		if props.EmbeddingDim > 0 {
			s.cfg.Embedding.Dimension = props.EmbeddingDim
		}
	}

	// Apply agent context window
	agentKey := s.cfg.Agent.Backend + ":" + s.cfg.Agent.Model
	if props, ok := probed[agentKey]; ok && props.ContextSize > 0 {
		s.cfg.Agent.ContextWindow = props.ContextSize
		if s.cfg.Agent.MaxTokens > props.ContextSize {
			s.logger.Warn().Int("max_tokens", s.cfg.Agent.MaxTokens).Int("context_window", props.ContextSize).
				Msg("agent max_tokens exceeds context window")
		}
	}

	// Apply reranker context window — used as the per-doc truncation budget
	// inside RerankClient. The /props probe loads the reranker model into the
	// router so the first /rerank call doesn't pay cold-load latency.
	rerankKey := s.cfg.Reranker.Backend + ":" + s.cfg.Reranker.Model
	if props, ok := probed[rerankKey]; ok && props.ContextSize > 0 {
		s.cfg.Reranker.MaxTokens = props.ContextSize
	}
}

// buildAgent constructs a fully-wired agent given a chat client. Used by both
// server.New (initial setup) and rebuildClients (config reload), so the
// search-tool wiring (store, max-images) and vision getter cannot drift
// between the two paths.
//
// Vision getter design: agent.vision.enabled and public_base_url are
// reported separately. The getter returns (false, "") when Enabled is off,
// and (true, "") when Enabled is on but PublicBaseURL is empty — the
// agent's switch distinguishes these two failure modes in its diagnostic
// logging.
func (s *Server) buildAgent(client *llm.ChatClient) *agent.Agent {
	searchTool := agent.NewSemanticSearchTool(s.pipeline)
	searchTool.SetStore(s.store)
	searchTool.SetMaxImages(s.cfg.Agent.Vision.MaxImagesPerCall)

	readDocTool := agent.NewReadDocumentTool(s.pipeline, s.store)
	readDocTool.SetDefaultTopK(s.cfg.Retrieval.TopK)

	// list_sources reads from the in-memory Registry, which the admin handler
	// updates atomically when the operator saves an edit. No file I/O on the
	// agent's per-turn hot path.
	listSourcesTool := agent.NewListSourcesTool(func() []domain.CrawlSource {
		if s.sources == nil {
			return nil
		}
		return s.sources.Snapshot()
	})

	tools := []agent.Tool{searchTool, readDocTool, listSourcesTool}

	systemPrompt := s.agentSystemPrompt()
	agentCounter, err := s.resolveCounter(s.cfg.Agent.Backend, s.cfg.Agent.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("agent token counter not available, tool output guards disabled")
	}

	a := agent.NewAgent(
		client, tools, systemPrompt,
		s.cfg.Agent.MaxIterations, agentCounter,
		s.cfg.Agent.ContextWindow, s.cfg.Agent.ToolOutputMaxTokens,
		s.sessionStore, s.logger,
	)
	a.SetVisionGetter(func() (bool, string) {
		if !s.cfg.Agent.Vision.Enabled {
			return false, ""
		}
		return true, s.cfg.Agent.Vision.PublicBaseURL
	})
	return a
}

// agentSystemPrompt renders the system prompt template with current data.
func (s *Server) agentSystemPrompt() string {
	ctx := context.Background()
	docCount, _ := s.store.CountDocuments(ctx)
	chunkCount, _ := s.store.CountChunks(ctx)

	rendered, err := prompts.RenderAgent(s.cfg.Agent.SystemPrompt, prompts.PromptData{
		DocumentCount: docCount,
		ChunkCount:    chunkCount,
	})
	if err != nil {
		s.logger.Warn().Err(err).Msg("failed to render system prompt template, using raw text")
		if s.cfg.Agent.SystemPrompt != "" {
			return s.cfg.Agent.SystemPrompt
		}
		return prompts.Agent
	}
	return rendered
}
