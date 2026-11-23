package server

import (
	"context"
	"fmt"
	"strings"
	"time"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/backend"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/tokencount"
)

// resolveEmbedClient builds the [embedding] stage client from the live config.
func (s *Server) resolveEmbedClient() (*llm.EmbedClient, error) {
	b := s.cfg.EnabledBackend(s.cfg.Embedding.Backend)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for embedding", s.cfg.Embedding.Backend)
	}
	s.logger.Info().Str("stage", "embedding").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewEmbedClient(llm.EmbedClientConfig{
		BaseURL:    b.BaseURL,
		APIKey:     b.APIKey,
		Model:      s.cfg.Embedding.Model,
		BatchSize:  s.cfg.Embedding.BatchSize,
		MaxRetries: b.MaxRetries,
	}), nil
}

// resolveChatClient builds the [agent] stage chat client from the live config.
func (s *Server) resolveChatClient() (*llm.ChatClient, error) {
	b := s.cfg.EnabledBackend(s.cfg.Agent.Backend)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for chat", s.cfg.Agent.Backend)
	}
	s.logger.Info().Str("stage", "agent").Str("backend", b.Name).Str("url", b.BaseURL).Msg("backend resolved")
	return llm.NewChatClient(llm.ChatClientConfig{
		BaseURL:     b.BaseURL,
		APIKey:      b.APIKey,
		Model:       s.cfg.Agent.Model,
		MaxTokens:   int64(s.cfg.Agent.MaxTokens),
		Temperature: s.cfg.Agent.Temperature,
		MaxRetries:  b.MaxRetries,
	}), nil
}

// resolveEmbedding resolves the embedding client and token counter from the
// live config and publishes both diag issues. The embedder is a true nil
// (not a typed-nil *EmbedClient) when resolution fails, so handle consumers'
// nil checks hold.
func (s *Server) resolveEmbedding() (llm.Embedder, tokencount.Counter) {
	var embedder llm.Embedder
	client, err := s.resolveEmbedClient()
	if err != nil {
		s.logger.Warn().Err(err).Msg("embedding backend not available")
	} else {
		embedder = client
	}
	s.publishProbeResult(issueEmbeddingBackend, err)

	counter, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("token counter not available (fix via admin UI)")
	}
	s.publishProbeResult(issueEmbeddingCounter, err)
	return embedder, counter
}

// resolveAgent builds the agent from the live config, or returns nil when no
// [agent] backend is configured or it cannot be resolved. It keeps the agent
// diag issues in step either way.
func (s *Server) resolveAgent() *agent.Agent {
	if s.cfg.Agent.Backend == "" {
		s.logger.Warn().Msg("no [agent] backend configured, agent disabled")
		s.diag.Set(issueAgentNoBackend)
		s.clearBudgetIssues()
		return nil
	}
	s.diag.Clear(issueAgentNoBackend.ID)

	client, err := s.resolveChatClient()
	s.publishProbeResult(issueAgentBackend, err)
	if err != nil {
		s.logger.Warn().Err(err).Msg("agent backend not available, agent disabled")
		s.clearBudgetIssues()
		return nil
	}
	a := s.buildAgent(client)
	s.logger.Info().Str("backend", s.cfg.Agent.Backend).
		Bool("vision_enabled", s.cfg.Agent.Vision.Enabled).
		Str("public_base_url", s.cfg.Agent.Vision.PublicBaseURL).
		Msg("agent initialized")
	return a
}

// resolveRerankClient resolves a backend by name and builds a reranker client.
// The reranker is llama.cpp-only because /rerank is a llama.cpp/Jina extension
// not part of the OpenAI surface — config.Validate enforces this at load time
// but we re-check here so a TOML edited on disk can't bypass the safeguard.
func (s *Server) resolveRerankClient(backendName string, partialCfg llm.RerankClientConfig) (*llm.RerankClient, error) {
	b := s.cfg.EnabledBackend(backendName)
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
		s.retriever.SetReranker(rag.RerankerSetup{})
		return
	}

	// Token counter is required for the context-budget guard; without it,
	// oversized docs would hit the cross-encoder's context window and the
	// /rerank call would fail at request time. Bail out if we can't resolve
	// one — failure here is a config issue worth surfacing.
	counter, err := s.resolveCounter(s.cfg.Reranker.Backend, s.cfg.Reranker.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("reranker disabled: no token counter")
		s.retriever.SetReranker(rag.RerankerSetup{})
		return
	}

	rerankMaxTokens := s.probed.RerankerMaxTokens()
	rc, err := s.resolveRerankClient(s.cfg.Reranker.Backend, llm.RerankClientConfig{
		Model:          s.cfg.Reranker.Model,
		Counter:        counter,
		MaxTokens:      rerankMaxTokens,
		ReservedTokens: s.cfg.Reranker.ReservedTokens,
		Timeout:        s.cfg.Reranker.Timeout,
	})
	if err != nil {
		s.logger.Warn().Err(err).Msg("reranker disabled: client resolution failed")
		s.retriever.SetReranker(rag.RerankerSetup{})
		return
	}

	s.retriever.SetReranker(rag.RerankerSetup{
		Reranker:    rc,
		Pool:        s.cfg.Reranker.TopN,
		TitleInText: s.cfg.Reranker.TitleInText,
		Logger:      s.logger,
	})
	s.logger.Info().
		Str("backend", s.cfg.Reranker.Backend).
		Str("model", s.cfg.Reranker.Model).
		Int("top_n", s.cfg.Reranker.TopN).
		Int("max_tokens", rerankMaxTokens).
		Msg("reranker enabled")
}

// resolveCounter creates an appropriate token counter for the given backend.
func (s *Server) resolveCounter(backendName, model string) (tokencount.Counter, error) {
	b := s.cfg.EnabledBackend(backendName)
	if b == nil {
		return nil, fmt.Errorf("no enabled backend named %q for token counter", backendName)
	}
	if b.IsLlamaCpp {
		base := strings.TrimSuffix(b.BaseURL, "/v1")
		s.logger.Info().Str("backend", b.Name).Msg("using llama.cpp token counter")
		counter := tokencount.NewLlamaCppCounter(base, model, nil)
		if b.MaxRetries != 0 {
			counter = counter.WithMaxRetries(b.MaxRetries)
		}
		return counter, nil
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
	}
	// The probe makes the router load the model; a disabled reranker must not
	// occupy the VRAM that disabling it is meant to free.
	if s.cfg.Reranker.Enabled {
		stages = append(stages, stageRef{"reranker", s.cfg.Reranker.Backend, s.cfg.Reranker.Model})
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
		b := s.cfg.EnabledBackend(st.backend)
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
			Int("total_slots", props.TotalSlots).
			Int("n_embd", props.EmbeddingDim).
			Str("model_path", props.ModelPath).
			Str("model_alias", props.ModelAlias).
			Bool("supports_vision_detected", props.SupportsVision).
			Msg("probed llama.cpp backend")

		// Vision sanity-check (agent stage only): the probe never overrides the
		// manual agent.vision.enabled flag — the operator's assertion is
		// authoritative — but it logs whenever its reading disagrees with the
		// flag, in either direction. Only meaningful for the agent stage;
		// embedding/reranker probes also pass through here.
		if st.name == "agent" {
			switch {
			case props.SupportsVision && !s.cfg.Agent.Vision.Enabled:
				// Capable model, flag off: suggest the operator turn it on in
				// admin without re-reading the discovery output.
				s.logger.Info().Str("backend", b.Name).Str("model", st.model).
					Msg("hint: probe detected modalities.vision — consider enabling agent.vision.enabled in admin")
			case !props.SupportsVision && s.cfg.Agent.Vision.Enabled:
				// Flag on, model can't see: image attachment will fail at
				// inference, so warn rather than wait for the runtime error.
				s.logger.Warn().Str("backend", b.Name).Str("model", st.model).
					Msg("warning: agent.vision.enabled is on but probe reports no modalities.vision — the configured model likely cannot process images")
			}
		}
	}

	// Apply probed results into the runtime holder (NOT s.cfg — config is the
	// fallback, not a write-back target; persisting probe residue destroyed the
	// "0 = auto" semantics). The reported n_ctx is already the per-request
	// limit the server enforces (see backend.LlamaCppProps.ContextSize) and is
	// taken verbatim; the embedding stage also records the vector dimension.
	embKey := s.cfg.Embedding.Backend + ":" + s.cfg.Embedding.Model
	if props, ok := probed[embKey]; ok {
		if props.ContextSize > 0 {
			s.probed.setEmbeddingMaxTokens(props.ContextSize)
		}
		if props.EmbeddingDim > 0 {
			s.probed.setEmbeddingDimension(props.EmbeddingDim)
		}
	}

	// Apply agent context window.
	agentKey := s.cfg.Agent.Backend + ":" + s.cfg.Agent.Model
	if props, ok := probed[agentKey]; ok {
		if props.ContextSize > 0 {
			s.probed.setAgentContextWindow(props.ContextSize)
			if s.cfg.Agent.MaxTokens > props.ContextSize {
				s.logger.Warn().Int("max_tokens", s.cfg.Agent.MaxTokens).Int("context_window", props.ContextSize).
					Msg("agent max_tokens exceeds probed context window")
			}
		}
	}

	// Apply reranker context window — used as the per-doc truncation budget
	// inside RerankClient. The /props probe loads the reranker model into the
	// router so the first /rerank call doesn't pay cold-load latency.
	rerankKey := s.cfg.Reranker.Backend + ":" + s.cfg.Reranker.Model
	if props, ok := probed[rerankKey]; ok {
		if props.ContextSize > 0 {
			s.probed.setRerankerMaxTokens(props.ContextSize)
		}
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
	searchTool := agent.NewSemanticSearchTool(s.retriever)
	searchTool.SetStore(s.store)
	searchTool.SetMaxImages(s.cfg.Agent.Vision.MaxImagesPerCall)

	readDocTool := agent.NewReadDocumentTool(s.retriever, s.store)
	readDocTool.SetDefaultTopK(s.cfg.Retrieval.TopK)

	tools := []agent.Tool{searchTool, readDocTool}

	systemPrompt := s.agentSystemPrompt()
	agentCounter, err := s.resolveCounter(s.cfg.Agent.Backend, s.cfg.Agent.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("agent token counter not available, tool output guards disabled")
	}

	// Derive the per-request token budget. ContextWindow is the probed effective
	// per-slot window (probe wins, config is fallback). Fixed is measured from
	// the system prompt + tool schemas; a measurement failure leaves Fixed at 0
	// so the guard still works off the reserves alone rather than disabling it.
	budget := agent.Budget{
		ContextWindow:  s.probed.AgentContextWindow(),
		Generation:     s.cfg.Agent.MaxTokens,
		UserReserve:    s.cfg.Agent.UserReserveTokens,
		HistoryReserve: s.cfg.Agent.HistoryReserveTokens,
	}
	if agentCounter != nil {
		// Bounded: the llama.cpp counter retries network errors with exponential
		// backoff (minutes against a dead backend), and buildAgent runs inside
		// the admin-save handler — an unbounded measurement would hang the save.
		mctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		fixed, ferr := agent.MeasureFixed(mctx, agentCounter, systemPrompt, tools)
		cancel()
		if ferr != nil {
			s.logger.Warn().Err(ferr).Msg("failed to measure fixed agent token cost, proceeding with Fixed=0")
		} else {
			budget.Fixed = fixed
		}
	}

	// Surface the token-budget invariants on the dashboard. Replayed here so
	// both server.New and rebuildClients publish from the same place; the
	// agent-disabled branches Clear these instead (see clearBudgetIssues).
	s.publishBudgetIssues(budget, s.cfg.Agent.ToolOutputMaxTokens)

	a := agent.NewAgent(
		client, tools, systemPrompt,
		s.cfg.Agent.MaxIterations, agentCounter,
		budget, s.cfg.Agent.ToolOutputMaxTokens,
		s.sessionStore, s.logger,
	)
	a.SetVisionGetter(s.agentVision)
	return a
}

// agentVision is the agent's VisionGetter. It reads the live config so admin
// toggles apply without rebuilding the agent.
func (s *Server) agentVision() (bool, string) {
	if !s.cfg.Agent.Vision.Enabled {
		return false, ""
	}
	return true, s.cfg.Agent.Vision.PublicBaseURL
}

// agentSystemPrompt renders the system prompt template with current data.
func (s *Server) agentSystemPrompt() string {
	ctx := context.Background()
	docCount, _ := s.store.CountDocuments(ctx)
	chunkCount, _ := s.store.CountChunks(ctx)

	rendered, err := prompts.RenderAgent(s.cfg.Agent.SystemPrompt, prompts.Data{
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
