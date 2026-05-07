package server

import (
	"etsi-rag/internal/chunking"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/llm"
)

// rebuildClients rebuilds all LLM clients from the live config and swaps them
// into pipeline, API handler, admin handler, and crawler.
// Called after every config save via the admin UI.
//
// Also reconciles the diag registry so any issue whose underlying subsystem
// is now healthy gets cleared, and any subsystem that broke between saves
// gets surfaced.
func (s *Server) rebuildClients() {
	s.logger.Info().Msg("rebuilding clients after config change")

	// Propagate non-LLM config changes
	s.pipeline.SetConfig(s.cfg.Retrieval)
	s.apiHandler.SetPassthrough(s.cfg.Agent.Passthrough)
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
		s.diag.Set(diag.Issue{
			ID: "embedding.backend", Component: "embedding", Severity: diag.SeverityError,
			Message: "Embedding backend not available", Detail: err.Error(),
			Hint: "Configure or fix the [embedding] backend at /admin/backends",
		})
	} else {
		s.pipeline.SetEmbedder(ec)
		s.apiHandler.SetEmbeddingClient(ec)
		s.diag.Clear("embedding.backend")
	}

	// Rebuild token counter and chunker
	counter, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("token counter rebuild failed (fix via admin UI)")
		s.diag.Set(diag.Issue{
			ID: "embedding.counter", Component: "embedding", Severity: diag.SeverityWarn,
			Message: "Embedding token counter not available", Detail: err.Error(),
			Hint: "Configure tokenization for the embedding backend at /admin/backends",
		})
	} else {
		s.diag.Clear("embedding.counter")
	}
	s.probeLlamaCppBackends()
	s.pipeline.SetCounter(counter)
	s.apiHandler.SetCounter(counter)
	s.pipeline.SetChunker(chunking.NewStructureAwareChunker(s.cfg.Chunking, counter, s.cfg.Embedding.MaxTokens))

	// Reranker — re-evaluate every save so toggling enabled takes effect.
	s.attachReranker()

	// Agent (single chat client for agent)
	if s.cfg.Agent.Backend != "" {
		s.diag.Clear("agent.no_backend")
		if ac, err := s.resolveChatClient(s.cfg.Agent.Backend, llm.ChatClientConfig{
			Model:       s.cfg.Agent.Model,
			MaxTokens:   int64(s.cfg.Agent.MaxTokens),
			Temperature: s.cfg.Agent.Temperature,
		}); err != nil {
			s.logger.Warn().Err(err).Msg("agent client rebuild failed")
			s.adminHandler.SetAgent(nil)
			s.apiHandler.SetAgent(nil)
			s.diag.Set(diag.Issue{
				ID: "agent.backend", Component: "agent", Severity: diag.SeverityError,
				Message: "Agent backend not available; chat disabled",
				Detail:  err.Error(),
				Hint:    "Fix the [agent] backend at /admin/backends or /admin/config",
			})
		} else {
			a := s.buildAgent(ac)
			s.adminHandler.SetAgent(a)
			s.apiHandler.SetAgent(a)
			s.diag.Clear("agent.backend")
			s.logger.Info().Str("backend", s.cfg.Agent.Backend).Msg("agent rebuilt")
		}
	} else {
		s.adminHandler.SetAgent(nil)
		s.apiHandler.SetAgent(nil)
		s.diag.Set(diag.Issue{
			ID: "agent.no_backend", Component: "agent", Severity: diag.SeverityError,
			Message: "No [agent] backend configured; chat disabled",
			Hint:    "Set [agent].backend at /admin/config",
		})
	}
}
