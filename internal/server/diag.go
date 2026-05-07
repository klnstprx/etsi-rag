package server

import (
	"context"
	"fmt"

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
)

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
