package server

import (
	"testing"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/diag"
)

// hasIssue reports whether the slice contains an issue with the given ID.
func hasIssue(issues []diag.Issue, id string) bool {
	for _, i := range issues {
		if i.ID == id {
			return true
		}
	}
	return false
}

func TestBudgetIssues(t *testing.T) {
	// A baseline budget with a healthy window: the tool pool is comfortably
	// larger than max_context_tokens and top_k chunks fit, so neither issue
	// should fire. Window 8192, gen 1024, fixed 1000, reserves 512+1280 →
	// pool = 8192-1024-1000-512-1280 = 4376.
	healthy := agent.Budget{
		ContextWindow:  8192,
		Generation:     1024,
		Fixed:          1000,
		UserReserve:    512,
		HistoryReserve: 1280,
	}

	tests := []struct {
		name       string
		budget     agent.Budget
		toolCap    int // ToolOutputMaxTokens override (0 = auto from pool)
		ret        config.RetrievalConfig
		chk        config.ChunkingConfig
		wantBudget bool // expect agent.budget
		wantTopK   bool // expect retrieval.topk_clip
	}{
		{
			name:       "neither fires",
			budget:     healthy,
			toolCap:    0,
			ret:        config.RetrievalConfig{TopK: 5, MaxContextTokens: 2048},
			chk:        config.ChunkingConfig{MaxChunkSize: 256},
			wantBudget: false,
			wantTopK:   false,
		},
		{
			name: "both fire",
			// Tiny pool so max_context_tokens > pool, and top_k×(chunk+24) far
			// exceeds the context budget. Window 2000, gen 1024, fixed 200,
			// reserves 512+1280 = pool = 2000-1024-200-512-1280 < 0 → clamped 0.
			budget: agent.Budget{
				ContextWindow:  2000,
				Generation:     1024,
				Fixed:          200,
				UserReserve:    512,
				HistoryReserve: 1280,
			},
			toolCap:    0,
			ret:        config.RetrievalConfig{TopK: 10, MaxContextTokens: 1024},
			chk:        config.ChunkingConfig{MaxChunkSize: 256},
			wantBudget: true, // 1024 > 0
			wantTopK:   true, // 10×(256+24)=2800 > 1024
		},
		{
			name: "budget fires alone",
			// Pool < max_context_tokens, but top_k chunks still fit the budget.
			// pool=4376; set max_context_tokens=5000 > pool. With top_k=1 and a
			// small chunk, 1×(100+24)=124 < 5000 so topk_clip does not fire.
			budget:     healthy,
			toolCap:    0,
			ret:        config.RetrievalConfig{TopK: 1, MaxContextTokens: 5000},
			chk:        config.ChunkingConfig{MaxChunkSize: 100},
			wantBudget: true,  // 5000 > 4376
			wantTopK:   false, // 124 < 5000
		},
		{
			name: "topk fires alone",
			// Pool comfortably covers max_context_tokens, but top_k×(chunk+24)
			// overflows the context budget.
			budget:     healthy,
			toolCap:    0,
			ret:        config.RetrievalConfig{TopK: 20, MaxContextTokens: 2048},
			chk:        config.ChunkingConfig{MaxChunkSize: 256},
			wantBudget: false, // 2048 < 4376
			wantTopK:   true,  // 20×(256+24)=5600 > 2048
		},
		{
			name: "explicit override path: cap larger than pool clears budget",
			// Pool is tiny (would fire on its own), but an explicit
			// ToolOutputMaxTokens override > max_context_tokens means a full
			// retrieval result fits, so agent.budget must NOT fire.
			budget: agent.Budget{
				ContextWindow:  2000,
				Generation:     1024,
				Fixed:          200,
				UserReserve:    512,
				HistoryReserve: 1280,
			},
			toolCap:    4096, // override > max_context_tokens
			ret:        config.RetrievalConfig{TopK: 1, MaxContextTokens: 2048},
			chk:        config.ChunkingConfig{MaxChunkSize: 100},
			wantBudget: false, // 2048 < 4096 (override wins over the clamped pool)
			wantTopK:   false, // 1×(100+24)=124 < 2048
		},
		{
			name: "explicit override path: cap smaller than max_context fires budget",
			// Healthy pool, but the operator pinned a small per-tool cap below
			// max_context_tokens — the override wins and agent.budget fires.
			budget:     healthy,
			toolCap:    512, // override < max_context_tokens
			ret:        config.RetrievalConfig{TopK: 1, MaxContextTokens: 2048},
			chk:        config.ChunkingConfig{MaxChunkSize: 100},
			wantBudget: true,  // 2048 > 512 (override, not pool, decides)
			wantTopK:   false, // 124 < 2048
		},
		{
			name: "zero ContextWindow skips both checks (probe failed)",
			// No probe and no config fallback → ContextWindow 0. Both checks are
			// meaningless without a known window, so return nil.
			budget: agent.Budget{
				ContextWindow:  0,
				Generation:     1024,
				Fixed:          200,
				UserReserve:    512,
				HistoryReserve: 1280,
			},
			toolCap:    0,
			ret:        config.RetrievalConfig{TopK: 20, MaxContextTokens: 5000},
			chk:        config.ChunkingConfig{MaxChunkSize: 256},
			wantBudget: false,
			wantTopK:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := budgetIssues(tt.budget, tt.toolCap, tt.ret, tt.chk)

			if hasIssue(got, "agent.budget") != tt.wantBudget {
				t.Errorf("agent.budget present = %v, want %v (issues=%+v)",
					hasIssue(got, "agent.budget"), tt.wantBudget, got)
			}
			if hasIssue(got, "retrieval.topk_clip") != tt.wantTopK {
				t.Errorf("retrieval.topk_clip present = %v, want %v (issues=%+v)",
					hasIssue(got, "retrieval.topk_clip"), tt.wantTopK, got)
			}

			// When ContextWindow is 0 the function must return nil outright.
			if tt.budget.ContextWindow == 0 && got != nil {
				t.Errorf("zero ContextWindow must return nil, got %+v", got)
			}

			// Verify the emitted issues carry the expected component/severity so
			// the dashboard renders them correctly.
			for _, i := range got {
				switch i.ID {
				case "agent.budget":
					if i.Component != "agent" {
						t.Errorf("agent.budget Component = %q, want agent", i.Component)
					}
					if i.Severity != diag.SeverityError {
						t.Errorf("agent.budget Severity = %v, want Error", i.Severity)
					}
					if i.Detail == "" {
						t.Error("agent.budget Detail should show the numbers")
					}
				case "retrieval.topk_clip":
					if i.Component != "retrieval" {
						t.Errorf("retrieval.topk_clip Component = %q, want retrieval", i.Component)
					}
					if i.Severity != diag.SeverityWarn {
						t.Errorf("retrieval.topk_clip Severity = %v, want Warn", i.Severity)
					}
					if i.Detail == "" {
						t.Error("retrieval.topk_clip Detail should show the clip count")
					}
				}
			}
		})
	}
}
