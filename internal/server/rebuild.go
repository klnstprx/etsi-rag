package server

import (
	"etsi-rag/internal/agent"
	"etsi-rag/internal/chunking"
	"etsi-rag/internal/llm"
)

// rebuildClients rebuilds all LLM clients from the live config and swaps them
// into pipeline, API handler, admin handler, and crawler.
// Called after every config save via the admin UI.
func (s *Server) rebuildClients() {
	s.logger.Info().Msg("rebuilding clients after config change")

	// Propagate non-LLM config changes
	s.pipeline.SetConfig(s.cfg.RAG)
	s.apiHandler.SetPassthrough(s.cfg.API.Passthrough)
	if s.crawler != nil {
		s.crawler.SetConfig(s.cfg.Crawler)
	}

	// Embedding
	if ec, err := s.resolveEmbedClient(s.cfg.Embedding.Backend, llm.EmbedClientConfig{
		Model:     s.cfg.Embedding.Model,
		BatchSize: s.cfg.Embedding.BatchSize,
	}); err != nil {
		s.logger.Warn().Err(err).Msg("embedding client rebuild failed")
		s.pipeline.SetEmbedder(nil)
		s.apiHandler.SetEmbeddingClient(nil)
	} else {
		s.pipeline.SetEmbedder(ec)
		s.apiHandler.SetEmbeddingClient(ec)
	}

	// Rebuild token counter and chunker
	counter, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("token counter rebuild failed (fix via admin UI)")
	}
	s.probeLlamaCppBackends()
	s.pipeline.SetCounter(counter)
	s.apiHandler.SetCounter(counter)
	s.pipeline.SetChunker(chunking.NewStructureAwareChunker(s.cfg.Chunking, counter, s.cfg.Embedding.MaxTokens))

	// Agent (single chat client for agent)
	if s.cfg.Agent.Backend != "" {
		if ac, err := s.resolveChatClient(s.cfg.Agent.Backend, llm.ChatClientConfig{
			Model:       s.cfg.Agent.Model,
			MaxTokens:   int64(s.cfg.Agent.MaxTokens),
			Temperature: s.cfg.Agent.Temperature,
		}); err != nil {
			s.logger.Warn().Err(err).Msg("agent client rebuild failed")
			s.adminHandler.SetAgent(nil)
			s.apiHandler.SetAgent(nil)
		} else {
			tools := []agent.Tool{
				agent.NewSemanticSearchTool(s.pipeline),
				agent.NewListDocumentsTool(s.store),
			}
			systemPrompt := s.agentSystemPrompt()
			agentCounter, counterErr := s.resolveCounter(s.cfg.Agent.Backend, s.cfg.Agent.Model)
			if counterErr != nil {
				s.logger.Warn().Err(counterErr).Msg("agent token counter rebuild failed, tool output guards disabled")
			}
			a := agent.NewAgent(ac, tools, systemPrompt, s.cfg.Agent.MaxIterations, agentCounter, s.cfg.Agent.ContextWindow, s.sessionStore, s.logger)
			s.adminHandler.SetAgent(a)
			s.apiHandler.SetAgent(a)
			s.logger.Info().Str("backend", s.cfg.Agent.Backend).Msg("agent rebuilt")
		}
	} else {
		s.adminHandler.SetAgent(nil)
		s.apiHandler.SetAgent(nil)
	}
}
