package server

import (
	"sync"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
)

// defaultEmbeddingMaxTokens is the safe embedding-input cap used when neither
// the probe nor the config supplies one. Matches the historical [embedding]
// max_tokens default: small enough that no embedding backend silently
// truncates a chunk.
const defaultEmbeddingMaxTokens = 512

// probedProps holds values auto-detected from llama.cpp /props at startup (and
// on every config reload). It is the runtime source of truth for physical
// model/server capacities — the agent's effective context window, the
// embedding model's max input tokens and vector dimension, and the reranker's
// context budget.
//
// Resolution rule for the capacity fields (context window, max tokens):
// an explicit config value (>0) is an operator override and wins; 0 means
// auto — follow the probe. This keeps the default zero-maintenance (server-side
// serving changes are picked up by the probe) while still letting an operator
// pin a value deliberately. Embedding dimension is the exception: it is a
// physical property that must match what the live model actually returns
// (a mismatch corrupts the vector store), so there a successful probe always
// wins and config is only the no-probe fallback.
//
// Probe results are NEVER written back to config (that destroyed the
// "0 = auto" semantics and persisted residue). All fields default to 0
// (= not probed). The holder pins the live *config.Config so getters observe
// admin edits without re-wiring.
type probedProps struct {
	cfg    *config.Config
	logger zerolog.Logger

	mu             sync.RWMutex
	agentCtx       int // probed per-request agent context window (slot n_ctx)
	embMaxTokens   int // probed embedding model max input tokens (slot n_ctx)
	embDim         int // probed embedding vector dimension (n_embd)
	rerankMaxToken int // probed reranker per-doc context budget (slot n_ctx)
}

// newProbedProps constructs a holder bound to the live config and logger.
func newProbedProps(cfg *config.Config, logger zerolog.Logger) *probedProps {
	return &probedProps{cfg: cfg, logger: logger}
}

// resolveCapacity applies the 0=auto/override rule and logs when an explicit
// override shadows a disagreeing probe (the operator should know the server
// reports a different capacity than they pinned).
func (p *probedProps) resolveCapacity(field string, probed, configured int) int {
	if configured > 0 {
		if probed > 0 && probed != configured {
			p.logger.Info().
				Str("field", field).
				Int("configured", configured).
				Int("probed", probed).
				Msg("explicit config value overrides probed capacity")
		}
		return configured
	}
	return probed
}

func (p *probedProps) setAgentContextWindow(v int) {
	p.mu.Lock()
	p.agentCtx = v
	p.mu.Unlock()
}

func (p *probedProps) AgentContextWindow() int {
	p.mu.RLock()
	v := p.agentCtx
	p.mu.RUnlock()
	return p.resolveCapacity("agent.context_window", v, p.cfg.Agent.ContextWindow)
}

func (p *probedProps) setEmbeddingMaxTokens(v int) {
	p.mu.Lock()
	p.embMaxTokens = v
	p.mu.Unlock()
}

func (p *probedProps) EmbeddingMaxTokens() int {
	p.mu.RLock()
	v := p.embMaxTokens
	p.mu.RUnlock()
	if got := p.resolveCapacity("embedding.max_tokens", v, p.cfg.Embedding.MaxTokens); got > 0 {
		return got
	}
	return defaultEmbeddingMaxTokens
}

func (p *probedProps) setEmbeddingDimension(v int) {
	p.mu.Lock()
	p.embDim = v
	p.mu.Unlock()
}

// EmbeddingDimension follows the probe when available (physical property —
// must match the vectors the embedder actually returns); config is only the
// no-probe fallback.
func (p *probedProps) EmbeddingDimension() int {
	p.mu.RLock()
	v := p.embDim
	p.mu.RUnlock()
	if v > 0 {
		if cfg := p.cfg.Embedding.Dimension; cfg > 0 && cfg != v {
			p.logger.Info().
				Str("field", "embedding.dimension").
				Int("probed", v).
				Int("configured", cfg).
				Msg("using probed embedding dimension; configured value ignored")
		}
		return v
	}
	return p.cfg.Embedding.Dimension
}

func (p *probedProps) setRerankerMaxTokens(v int) {
	p.mu.Lock()
	p.rerankMaxToken = v
	p.mu.Unlock()
}

func (p *probedProps) RerankerMaxTokens() int {
	p.mu.RLock()
	v := p.rerankMaxToken
	p.mu.RUnlock()
	return p.resolveCapacity("reranker.max_tokens", v, p.cfg.Reranker.MaxTokens)
}
