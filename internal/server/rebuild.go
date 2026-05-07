package server

import (
	"etsi-rag/internal/chunking"
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
	embedName, embedCfg := s.embedClientConfig()
	ec, err := s.resolveEmbedClient(embedName, embedCfg)
	if err != nil {
		s.logger.Warn().Err(err).Msg("embedding client rebuild failed")
		s.pipeline.SetEmbedder(nil)
		s.apiHandler.SetEmbeddingClient(nil)
	} else {
		s.pipeline.SetEmbedder(ec)
		s.apiHandler.SetEmbeddingClient(ec)
	}
	s.publishProbeResult(issueEmbeddingBackend, err)

	// Rebuild token counter and chunker
	counter, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("token counter rebuild failed (fix via admin UI)")
	}
	s.publishProbeResult(issueEmbeddingCounter, err)
	s.probeLlamaCppBackends()
	s.pipeline.SetCounter(counter)
	s.apiHandler.SetCounter(counter)
	s.pipeline.SetChunker(chunking.NewStructureAwareChunker(s.cfg.Chunking, counter, s.cfg.Embedding.MaxTokens))

	// Reranker — re-evaluate every save so toggling enabled takes effect.
	s.attachReranker()

	// Agent (single chat client for agent)
	if s.cfg.Agent.Backend != "" {
		s.diag.Clear(issueAgentNoBackend.ID)
		chatName, chatCfg := s.chatClientConfig()
		ac, err := s.resolveChatClient(chatName, chatCfg)
		if err != nil {
			s.logger.Warn().Err(err).Msg("agent client rebuild failed")
			s.adminHandler.SetAgent(nil)
			s.apiHandler.SetAgent(nil)
		} else {
			a := s.buildAgent(ac)
			s.adminHandler.SetAgent(a)
			s.apiHandler.SetAgent(a)
			s.logger.Info().Str("backend", s.cfg.Agent.Backend).Msg("agent rebuilt")
		}
		s.publishProbeResult(issueAgentBackend, err)
	} else {
		s.adminHandler.SetAgent(nil)
		s.apiHandler.SetAgent(nil)
		s.diag.Set(issueAgentNoBackend)
	}
}
