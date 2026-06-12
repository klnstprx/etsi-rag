package server

import (
	"etsi-rag/internal/chunking"
)

// rebuildClients rebuilds all LLM clients from the live config and routes
// them through the shared handles + retriever/ingester references.
// Called after every config save via the admin UI.
//
// Single source of truth: hot-swapped runtime resources (embedder, agent)
// live on Handle types. One Set call updates the handle; every consumer
// (retriever, ingester, api, admin) picks up the new client on its next Get.
// No parallel fanout, no risk of partial swaps leaving consumers out of sync.
//
// Also reconciles the diag registry so any issue whose underlying subsystem
// is now healthy gets cleared, and any subsystem that broke between saves
// gets surfaced.
func (s *Server) rebuildClients() {
	s.logger.Info().Msg("rebuilding clients after config change")

	// Propagate non-LLM config changes
	s.retriever.SetConfig(s.cfg.Retrieval)
	if s.crawler != nil {
		s.crawler.SetConfig(s.cfg.Crawler)
	}

	// Embedding — single Set propagates to retriever + ingester (and anywhere
	// else that holds *llm.EmbedderHandle in the future).
	embedName, embedCfg := s.embedClientConfig()
	ec, err := s.resolveEmbedClient(embedName, embedCfg)
	if err != nil {
		s.logger.Warn().Err(err).Msg("embedding client rebuild failed")
		s.embedderHandle.Set(nil)
	} else {
		s.embedderHandle.Set(ec)
	}
	s.publishProbeResult(issueEmbeddingBackend, err)

	// Token counter (retriever's BuildContext) and chunker (ingester).
	counter, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
	if err != nil {
		s.logger.Warn().Err(err).Msg("token counter rebuild failed (fix via admin UI)")
	}
	s.publishProbeResult(issueEmbeddingCounter, err)
	s.probeLlamaCppBackends()
	s.retriever.SetCounter(counter)
	s.ingester.SetChunker(chunking.NewStructureAwareChunker(s.cfg.Chunking, counter, s.probed.EmbeddingMaxTokens()))

	// Reranker — re-evaluate every save so toggling enabled takes effect.
	s.attachReranker()

	// Agent — one handle swap covers api and admin.
	if s.cfg.Agent.Backend != "" {
		s.diag.Clear(issueAgentNoBackend.ID)
		chatName, chatCfg := s.chatClientConfig()
		ac, err := s.resolveChatClient(chatName, chatCfg)
		if err != nil {
			s.logger.Warn().Err(err).Msg("agent client rebuild failed")
			s.agentHandle.Set(nil)
			// No agent → its budget can't be evaluated; drop stale invariants.
			s.clearBudgetIssues()
		} else {
			s.agentHandle.Set(s.buildAgent(ac))
			s.logger.Info().Str("backend", s.cfg.Agent.Backend).Msg("agent rebuilt")
		}
		s.publishProbeResult(issueAgentBackend, err)
	} else {
		s.agentHandle.Set(nil)
		s.diag.Set(issueAgentNoBackend)
		s.clearBudgetIssues()
	}
}
