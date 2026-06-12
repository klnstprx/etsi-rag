package server

import (
	"context"
	"fmt"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/sources"
)

// Diag issue templates owned by the server package. Single source of truth so
// the boot-time publish in server.New, the rebuild-time republish in
// rebuildClients, and the recheck registration in registerDiagRechecks all
// agree on Component / Severity / Message / Hint. Detail is filled in per-call
// from the underlying err.
//
// RequiresRestart issues (embedding.warm, objectstore.unavailable,
// crawler.disabled) are inlined at their single call site — they never recur.
var (
	issueEmbeddingBackend = diag.Issue{
		ID: "embedding.backend", Component: "embedding", Severity: diag.SeverityError,
		Message: "Embedding backend not available",
		Hint:    "Configure or fix the [embedding] backend at /admin/backends",
	}
	issueEmbeddingCounter = diag.Issue{
		ID: "embedding.counter", Component: "embedding", Severity: diag.SeverityWarn,
		Message: "Embedding token counter not available",
		Hint:    "Configure tokenization for the embedding backend at /admin/backends",
	}
	issueAgentBackend = diag.Issue{
		ID: "agent.backend", Component: "agent", Severity: diag.SeverityError,
		Message: "Agent backend not available; chat disabled",
		Hint:    "Fix the [agent] backend at /admin/backends or /admin/config",
	}
	issueAgentNoBackend = diag.Issue{
		ID: "agent.no_backend", Component: "agent", Severity: diag.SeverityError,
		Message: "No [agent] backend configured; chat disabled",
		Hint:    "Set [agent].backend at /admin/config",
	}
	issueSourcesReconcile = diag.Issue{
		ID: "sources.reconcile", Component: "sources", Severity: diag.SeverityWarn,
		Message: "Failed to reconcile orphan crawl pages",
		Hint:    "Check the SQLite store; non-fatal but orphans accumulate.",
	}
	issueAgentBudget = diag.Issue{
		ID: "agent.budget", Component: "agent", Severity: diag.SeverityError,
		Message: "Agent tool budget cannot fit a full retrieval result",
		Hint:    "Lower retrieval.max_context_tokens or agent.max_tokens, or serve the model with a larger context (raise the backend's n_ctx).",
	}
	issueRetrievalTopKClip = diag.Issue{
		ID: "retrieval.topk_clip", Component: "retrieval", Severity: diag.SeverityWarn,
		Message: "Retrieval top_k cannot fit in the context budget",
		Hint:    "Lower retrieval.top_k or chunking.max_chunk_size, or raise retrieval.max_context_tokens.",
	}
)

// chunkHeaderOverhead is the per-chunk framing BuildContext prepends to each
// retrieved chunk — the "[N] Title > Section:\n" header plus the trailing
// blank line. Used to estimate how many of top_k chunks can actually reach the
// LLM within the context budget.
const chunkHeaderOverhead = 24

// budgetIssues derives the agent/retrieval budget invariants as diag.Issues.
// Pure: it reads only its arguments so it can be unit-tested in isolation and
// replayed from both server.New and rebuildClients.
//
//   - agent.budget (error): fires when retrieval.max_context_tokens exceeds the
//     effective per-tool output cap. The cap is the explicit
//     agent.tool_output_max_tokens when > 0, otherwise the budget's ToolPool.
//     A full retrieval result handed to the agent would be truncated below the
//     budget the operator configured.
//   - retrieval.topk_clip (warn): fires when top_k chunks, each framed with
//     chunkHeaderOverhead, cannot all fit in max_context_tokens — so only some
//     of the retrieved chunks reach the LLM.
//
// When ContextWindow is 0 the per-request window is unknown (probe failed and
// no config fallback), so both checks are meaningless and the function returns
// nil.
func budgetIssues(b agent.Budget, toolOutputOverride int, ret config.RetrievalConfig, chk config.ChunkingConfig) []diag.Issue {
	if b.ContextWindow == 0 {
		return nil
	}

	var issues []diag.Issue

	effectiveToolCap := b.ToolPool()
	if toolOutputOverride > 0 {
		effectiveToolCap = toolOutputOverride
	}
	if ret.MaxContextTokens > effectiveToolCap {
		issue := issueAgentBudget
		issue.Detail = fmt.Sprintf(
			"window=%d, generation=%d, fixed=%d, user_reserve=%d, history_reserve=%d → tool pool=%d (effective cap=%d) < retrieval.max_context_tokens=%d",
			b.ContextWindow, b.Generation, b.Fixed, b.UserReserve, b.HistoryReserve,
			b.ToolPool(), effectiveToolCap, ret.MaxContextTokens,
		)
		issues = append(issues, issue)
	}

	perChunk := chk.MaxChunkSize + chunkHeaderOverhead
	if perChunk > 0 && ret.TopK*perChunk > ret.MaxContextTokens {
		fit := ret.MaxContextTokens / perChunk
		issue := issueRetrievalTopKClip
		issue.Detail = fmt.Sprintf(
			"only ~%d of top_k %d chunks can reach the LLM (max_context_tokens=%d / (max_chunk_size=%d + overhead=%d))",
			fit, ret.TopK, ret.MaxContextTokens, chk.MaxChunkSize, chunkHeaderOverhead,
		)
		issues = append(issues, issue)
	}

	return issues
}

// publishBudgetIssues replays the budget invariants onto the diag registry:
// any returned issue is Set, and any ID not present in the result is Cleared.
// Called from the buildAgent path (server.New + rebuildClients) so a config
// reload that fixes (or breaks) the budget is reflected without a restart.
func (s *Server) publishBudgetIssues(b agent.Budget, toolOutputOverride int) {
	got := budgetIssues(b, toolOutputOverride, s.cfg.Retrieval, s.cfg.Chunking)
	present := make(map[string]bool, len(got))
	for _, i := range got {
		s.diag.Set(i)
		present[i.ID] = true
	}
	for _, id := range []string{issueAgentBudget.ID, issueRetrievalTopKClip.ID} {
		if !present[id] {
			s.diag.Clear(id)
		}
	}
}

// clearBudgetIssues drops both budget invariants. Called on the agent-disabled
// branches (no backend configured, or the chat-client rebuild failed) so a
// stale budget error doesn't outlive a removed agent backend.
func (s *Server) clearBudgetIssues() {
	s.diag.Clear(issueAgentBudget.ID)
	s.diag.Clear(issueRetrievalTopKClip.ID)
}

// publishProbeResult sets or clears the issue defined by template based on err.
// nil err → Clear; non-nil → Set with template + Detail = err.Error().
func (s *Server) publishProbeResult(template diag.Issue, err error) {
	if err == nil {
		s.diag.Clear(template.ID)
		return
	}
	issue := template
	issue.Detail = err.Error()
	s.diag.Set(issue)
}

// registerDiagRechecks attaches per-issue recheck closures to the diag
// registry. Called once at boot, after all subsystems are constructed. The
// closures capture *Server, so they always operate on live config.
//
// RequiresRestart issues (objectstore.unavailable, embedding.warm,
// crawler.disabled) are NOT registered — there is no in-process probe that
// would meaningfully re-run them, and Recheck(no-args) skips them anyway.
func (s *Server) registerDiagRechecks() {
	s.diag.Register(issueEmbeddingBackend.ID, issueEmbeddingBackend, func(ctx context.Context) error {
		name, cfg := s.embedClientConfig()
		_, err := s.resolveEmbedClient(name, cfg)
		return err
	})

	s.diag.Register(issueEmbeddingCounter.ID, issueEmbeddingCounter, func(ctx context.Context) error {
		_, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
		return err
	})

	s.diag.Register(issueAgentBackend.ID, issueAgentBackend, func(ctx context.Context) error {
		// agent.no_backend covers the empty-string case; don't double-report.
		if s.cfg.Agent.Backend == "" {
			return nil
		}
		name, cfg := s.chatClientConfig()
		_, err := s.resolveChatClient(name, cfg)
		return err
	})

	s.diag.Register(issueAgentNoBackend.ID, issueAgentNoBackend, func(ctx context.Context) error {
		if s.cfg.Agent.Backend == "" {
			return fmt.Errorf("agent.backend is empty")
		}
		return nil
	})

	// Sources rechecks: re-load from disk so a hand-edit is picked up without
	// a server restart. The Registry's publish* hooks update the diag state.
	s.diag.Register(sources.IssueParse, sources.IssueParseTemplate, func(ctx context.Context) error {
		return s.sources.Load()
	})

	s.diag.Register(sources.IssueEmpty, sources.IssueEmptyTemplate, func(ctx context.Context) error {
		if err := s.sources.Load(); err != nil {
			return err
		}
		if len(s.sources.Snapshot()) == 0 {
			return fmt.Errorf("no sources defined in %s", s.sources.Path())
		}
		return nil
	})

	s.diag.Register(issueSourcesReconcile.ID, issueSourcesReconcile, func(ctx context.Context) error {
		_, err := s.store.MarkOrphanPagesStale(ctx, s.sources.KnownKeys())
		return err
	})
}
