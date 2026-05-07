package server

import (
	"context"
	"fmt"

	"etsi-rag/internal/diag"
	"etsi-rag/internal/llm"
)

// registerDiagRechecks attaches per-issue recheck closures to the diag
// registry. Called once at boot, after all subsystems are constructed. The
// closures capture *Server, so they always operate on live config (no
// snapshotting at registration time).
//
// RequiresRestart issues (objectstore.unavailable, embedding.warm,
// crawler.disabled) are NOT registered — there is no in-process probe that
// would meaningfully re-run them, and Recheck(no-args) skips them anyway.
// Restart is the operator's only recourse, and the dashboard says so.
func (s *Server) registerDiagRechecks() {
	s.diag.Register("embedding.backend",
		diag.Issue{
			ID: "embedding.backend", Component: "embedding", Severity: diag.SeverityError,
			Message: "Embedding backend not available",
			Hint:    "Configure or fix the [embedding] backend at /admin/backends",
		},
		func(ctx context.Context) error {
			_, err := s.resolveEmbedClient(s.cfg.Embedding.Backend, llm.EmbedClientConfig{
				Model:     s.cfg.Embedding.Model,
				BatchSize: s.cfg.Embedding.BatchSize,
			})
			return err
		},
	)

	s.diag.Register("embedding.counter",
		diag.Issue{
			ID: "embedding.counter", Component: "embedding", Severity: diag.SeverityWarn,
			Message: "Embedding token counter not available",
			Hint:    "Configure tokenization for the embedding backend at /admin/backends",
		},
		func(ctx context.Context) error {
			_, err := s.resolveCounter(s.cfg.Embedding.Backend, s.cfg.Embedding.Model)
			return err
		},
	)

	s.diag.Register("agent.backend",
		diag.Issue{
			ID: "agent.backend", Component: "agent", Severity: diag.SeverityError,
			Message: "Agent backend not available; chat disabled",
			Hint:    "Fix the [agent] backend at /admin/backends or /admin/config",
		},
		func(ctx context.Context) error {
			if s.cfg.Agent.Backend == "" {
				// agent.no_backend covers this case; don't double-report.
				return nil
			}
			_, err := s.resolveChatClient(s.cfg.Agent.Backend, llm.ChatClientConfig{
				Model:       s.cfg.Agent.Model,
				MaxTokens:   int64(s.cfg.Agent.MaxTokens),
				Temperature: s.cfg.Agent.Temperature,
			})
			return err
		},
	)

	s.diag.Register("agent.no_backend",
		diag.Issue{
			ID: "agent.no_backend", Component: "agent", Severity: diag.SeverityError,
			Message: "No [agent] backend configured; chat disabled",
			Hint:    "Set [agent].backend at /admin/config",
		},
		func(ctx context.Context) error {
			if s.cfg.Agent.Backend == "" {
				return fmt.Errorf("agent.backend is empty")
			}
			return nil
		},
	)

	// Sources rechecks: re-load the file from disk, then run reconciliation.
	// Lets a hand-edit be picked up without a server restart.
	s.diag.Register("sources.parse",
		diag.Issue{
			ID: "sources.parse", Component: "sources", Severity: diag.SeverityError,
			Message: "sources.toml failed to load",
			Hint:    "Edit sources.toml and re-check, or fix at /admin/crawler",
		},
		func(ctx context.Context) error {
			return s.sources.Load()
		},
	)

	s.diag.Register("sources.empty",
		diag.Issue{
			ID: "sources.empty", Component: "sources", Severity: diag.SeverityWarn,
			Message: "No crawl sources configured",
			Hint:    "Add a source at /admin/crawler",
		},
		func(ctx context.Context) error {
			// sources.Registry.Load already publishes IssueEmpty based on the
			// snapshot length, so triggering Load is enough here.
			if err := s.sources.Load(); err != nil {
				return err
			}
			if len(s.sources.Snapshot()) == 0 {
				return fmt.Errorf("no sources defined in %s", s.sources.Path())
			}
			return nil
		},
	)

	s.diag.Register("sources.reconcile",
		diag.Issue{
			ID: "sources.reconcile", Component: "sources", Severity: diag.SeverityWarn,
			Message: "Failed to reconcile orphan crawl pages",
			Hint:    "Check the SQLite store; non-fatal but orphans accumulate.",
		},
		func(ctx context.Context) error {
			_, err := s.store.MarkOrphanPagesStale(ctx, s.sources.KnownKeys())
			return err
		},
	)
}
