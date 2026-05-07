package admin

import (
	"net/http"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/diag"
)

// snapshotIssues returns a sorted snapshot of the diag registry, or nil when
// the registry isn't wired (e.g. unit tests that exercise the handler in
// isolation). Templates handle the nil/empty case with a "no issues" empty
// state.
func (h *Handler) snapshotIssues() []diag.Issue {
	if h.diag == nil {
		return nil
	}
	return h.diag.Snapshot()
}

// HealthIssues handles GET /admin/health/issues — returns the IssuesCard
// fragment for HTMX swap.
func (h *Handler) HealthIssues(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.IssuesCard(h.snapshotIssues()).Render(r.Context(), w)
}

// HealthRecheck handles POST /admin/health/recheck — re-runs every recheck
// whose issue is NOT RequiresRestart, then returns the refreshed IssuesCard.
// RequiresRestart issues stay set; their card row already says "Restart
// required" so the operator knows the button can't help them.
func (h *Handler) HealthRecheck(w http.ResponseWriter, r *http.Request) {
	if h.diag != nil {
		_ = h.diag.Recheck(r.Context())
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.IssuesCard(h.snapshotIssues()).Render(r.Context(), w)
}
