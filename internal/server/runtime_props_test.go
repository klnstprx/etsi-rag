package server

import (
	"testing"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
)

// TestProbedProps_ResolutionPrecedence locks the resolution rule for the
// capacity fields (agent context window, embedding/reranker max tokens):
// an explicit config value (>0) is an operator override and wins; 0 means
// auto — follow the probe. Embedding dimension is the exception: it is a
// physical property that must match what the live model actually returns,
// so a successful probe always wins there.
func TestProbedProps_ResolutionPrecedence(t *testing.T) {
	tests := []struct {
		name string
		// probed values (0 = not probed)
		agentCtx, embMax, embDim, rerankMax int
		// config values (0 = auto)
		cfgAgentCtx, cfgEmbMax, cfgEmbDim, cfgRerankMax     int
		wantAgentCtx, wantEmbMax, wantEmbDim, wantRerankMax int
	}{
		{
			name:     "explicit config overrides probe (capacities); probe wins for dimension",
			agentCtx: 16384, embMax: 4096, embDim: 2560, rerankMax: 8192,
			cfgAgentCtx: 32768, cfgEmbMax: 10240, cfgEmbDim: 4, cfgRerankMax: 2048,
			wantAgentCtx: 32768, wantEmbMax: 10240, wantEmbDim: 2560, wantRerankMax: 2048,
		},
		{
			name:     "zero config = auto, follows probe",
			agentCtx: 16384, embMax: 4096, embDim: 2560, rerankMax: 8192,
			cfgAgentCtx: 0, cfgEmbMax: 0, cfgEmbDim: 0, cfgRerankMax: 0,
			wantAgentCtx: 16384, wantEmbMax: 4096, wantEmbDim: 2560, wantRerankMax: 8192,
		},
		{
			name:     "no probe falls back to config",
			agentCtx: 0, embMax: 0, embDim: 0, rerankMax: 0,
			cfgAgentCtx: 4096, cfgEmbMax: 2048, cfgEmbDim: 2560, cfgRerankMax: 1024,
			wantAgentCtx: 4096, wantEmbMax: 2048, wantEmbDim: 2560, wantRerankMax: 1024,
		},
		{
			name:     "neither probe nor config: embedding max tokens keeps its safe default",
			agentCtx: 0, embMax: 0, embDim: 0, rerankMax: 0,
			cfgAgentCtx: 0, cfgEmbMax: 0, cfgEmbDim: 0, cfgRerankMax: 0,
			wantAgentCtx: 0, wantEmbMax: defaultEmbeddingMaxTokens, wantEmbDim: 0, wantRerankMax: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{}
			cfg.Agent.ContextWindow = tt.cfgAgentCtx
			cfg.Embedding.MaxTokens = tt.cfgEmbMax
			cfg.Embedding.Dimension = tt.cfgEmbDim
			cfg.Reranker.MaxTokens = tt.cfgRerankMax

			pp := newProbedProps(cfg, zerolog.Nop())
			pp.setAgentContextWindow(tt.agentCtx)
			pp.setEmbeddingMaxTokens(tt.embMax)
			pp.setEmbeddingDimension(tt.embDim)
			pp.setRerankerMaxTokens(tt.rerankMax)

			if got := pp.AgentContextWindow(); got != tt.wantAgentCtx {
				t.Errorf("AgentContextWindow() = %d, want %d", got, tt.wantAgentCtx)
			}
			if got := pp.EmbeddingMaxTokens(); got != tt.wantEmbMax {
				t.Errorf("EmbeddingMaxTokens() = %d, want %d", got, tt.wantEmbMax)
			}
			if got := pp.EmbeddingDimension(); got != tt.wantEmbDim {
				t.Errorf("EmbeddingDimension() = %d, want %d", got, tt.wantEmbDim)
			}
			if got := pp.RerankerMaxTokens(); got != tt.wantRerankMax {
				t.Errorf("RerankerMaxTokens() = %d, want %d", got, tt.wantRerankMax)
			}
		})
	}
}

// TestProbedProps_LiveConfig confirms getters read the LIVE config pointer:
// a config edit after construction is observed immediately, both as an
// override appearing (capacity) and as a fallback (dimension, unprobed).
func TestProbedProps_LiveConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.Embedding.Dimension = 4
	pp := newProbedProps(cfg, zerolog.Nop())

	if got := pp.EmbeddingDimension(); got != 4 {
		t.Fatalf("EmbeddingDimension() = %d, want 4 (config fallback)", got)
	}
	cfg.Embedding.Dimension = 128
	if got := pp.EmbeddingDimension(); got != 128 {
		t.Errorf("EmbeddingDimension() = %d, want 128 after live config edit", got)
	}
	// Once probed, dimension follows the probe regardless of config.
	pp.setEmbeddingDimension(2560)
	if got := pp.EmbeddingDimension(); got != 2560 {
		t.Errorf("EmbeddingDimension() = %d, want 2560 (probe wins for dimension)", got)
	}

	// Capacity field: probed value applies while config is 0 (auto)…
	pp.setAgentContextWindow(16384)
	if got := pp.AgentContextWindow(); got != 16384 {
		t.Errorf("AgentContextWindow() = %d, want 16384 (auto follows probe)", got)
	}
	// …and a live config edit to non-zero becomes an override on the next read.
	cfg.Agent.ContextWindow = 32768
	if got := pp.AgentContextWindow(); got != 32768 {
		t.Errorf("AgentContextWindow() = %d, want 32768 (explicit override)", got)
	}
}
