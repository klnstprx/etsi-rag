package server

import (
	"etsi-rag/internal/chunking"
)

// rebuildClients rebuilds all LLM clients from the live config after every
// admin save, through the same resolve* steps New uses, so diag issues clear
// or appear as subsystems recover or break.
func (s *Server) rebuildClients() {
	s.logger.Info().Msg("rebuilding clients after config change")

	// Propagate non-LLM config changes
	s.retriever.SetConfig(s.cfg.Retrieval)
	if s.crawler != nil {
		s.crawler.SetConfig(s.cfg.Crawler)
	}

	// One Set per handle reaches every consumer (retriever, ingester, api).
	embedder, counter := s.resolveEmbedding()
	s.embedderHandle.Set(embedder)
	s.probeLlamaCppBackends()
	s.retriever.SetCounter(counter)
	s.ingester.SetChunker(chunking.NewStructureAwareChunker(s.cfg.Chunking, counter, s.probed.EmbeddingMaxTokens()))
	s.ingester.SetTitleInSearch(s.cfg.Chunking.TitleInSearchText)

	// Reranker — re-evaluate every save so toggling enabled takes effect.
	s.attachReranker()

	s.agentHandle.Set(s.resolveAgent())
}
