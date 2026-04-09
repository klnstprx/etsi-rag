package server

import (
	"context"
	"fmt"
	"strings"

	"etsi-rag/internal/backend"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/tokencount"
)

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
			Msg("probed llama.cpp backend")
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
