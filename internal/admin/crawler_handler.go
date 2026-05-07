package admin

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"unicode"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/crawler"
	"etsi-rag/internal/domain"
)

// validateSourceKey rejects keys that are empty, contain a path separator, or
// contain non-printable runes. Belt-and-braces: sources.toml validation already
// covers stored keys, but route handlers receive their key from the URL path,
// which is operator-controlled.
func validateSourceKey(key string) error {
	trimmed := strings.TrimSpace(key)
	if trimmed == "" {
		return fmt.Errorf("key is required")
	}
	if trimmed != key {
		return fmt.Errorf("key must not have leading or trailing whitespace")
	}
	for _, r := range key {
		if r == '/' || !unicode.IsPrint(r) {
			return fmt.Errorf("key contains invalid character %q", r)
		}
	}
	return nil
}

// parseTags splits a comma-separated form value into a clean tag slice.
// Whitespace-only entries are dropped; the validator enforces the rest.
func parseTags(formValue string) []string {
	if formValue == "" {
		return nil
	}
	parts := strings.Split(formValue, ",")
	tags := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			tags = append(tags, t)
		}
	}
	if len(tags) == 0 {
		return nil
	}
	return tags
}

func (h *Handler) crawlerData(ctx context.Context) templates.CrawlerData {
	running := h.crawler != nil && h.crawler.IsRunning()

	docCount, _ := h.store.CountDocuments(ctx)
	var sources []domain.CrawlSource
	if h.sources != nil {
		sources = h.sources.Snapshot()
	}
	runs, _ := h.store.ListCrawlRuns(ctx, 10)

	var progress *templates.ProgressSnapshot
	if running && h.crawler != nil {
		progress = progressSnapshotFrom(h.crawler.Progress())
	}

	return templates.CrawlerData{
		Running:  running,
		DocCount: docCount,
		Sources:  sources,
		Runs:     runs,
		Progress: progress,
	}
}

// CrawlerPage handles GET /admin/crawler.
func (h *Handler) CrawlerPage(w http.ResponseWriter, r *http.Request) {
	data := h.crawlerData(r.Context())
	page := templates.CrawlerPage(data)
	content := templates.CrawlerContent(data)
	render(w, r, page, content)
}

// crawlerGuard checks common preconditions for crawler actions (configured, not running,
// pipeline available and reachable). Returns true if the guard failed and a response was sent.
func (h *Handler) crawlerGuard(w http.ResponseWriter, r *http.Request, label string) bool {
	if h.crawler == nil {
		http.Error(w, "crawler not configured", http.StatusServiceUnavailable)
		return true
	}

	var msg string
	var level string
	switch {
	case h.crawler.IsRunning():
		msg, level = "Crawl already running", "warning"
	case h.pipeline == nil:
		msg, level = "Cannot "+label+": pipeline not configured", "error"
	default:
		if err := h.pipeline.Preflight(r.Context()); err != nil {
			msg, level = "Cannot "+label+": "+err.Error(), "error"
		}
	}

	if msg != "" {
		h.renderCrawler(w, r, msg, level)
		return true
	}
	return false
}

// CrawlerRun handles POST /admin/crawler/run.
func (h *Handler) CrawlerRun(w http.ResponseWriter, r *http.Request) {
	if h.crawlerGuard(w, r, "start crawl") {
		return
	}

	go func() {
		runID, err := h.crawler.RunAll(h.baseCtx)
		if err != nil {
			h.logger.Error().Err(err).Int64("run_id", runID).Msg("crawl failed")
		} else {
			h.logger.Info().Int64("run_id", runID).Msg("crawl completed")
		}
	}()

	h.logger.Info().Msg("crawler started")
	h.renderCrawler(w, r, "Crawl started", "success")
}

// CrawlerReprocess handles POST /admin/crawler/reprocess.
func (h *Handler) CrawlerReprocess(w http.ResponseWriter, r *http.Request) {
	if h.crawlerGuard(w, r, "reprocess") {
		return
	}

	go func() {
		runID, err := h.crawler.Reprocess(h.baseCtx)
		if err != nil {
			h.logger.Error().Err(err).Int64("run_id", runID).Msg("reprocess failed")
		} else {
			h.logger.Info().Int64("run_id", runID).Msg("reprocess completed")
		}
	}()

	h.logger.Info().Msg("reprocess started")
	h.renderCrawler(w, r, "Reprocess started", "success")
}

// CrawlerStop handles DELETE /admin/crawler/run.
func (h *Handler) CrawlerStop(w http.ResponseWriter, r *http.Request) {
	if h.crawler != nil {
		h.crawler.Cancel()
	}
	h.renderCrawler(w, r, "Crawl stop requested", "success")
}

// CrawlerStatus handles GET /admin/crawler/status (polled by HTMX).
func (h *Handler) CrawlerStatus(w http.ResponseWriter, r *http.Request) {
	running := h.crawler != nil && h.crawler.IsRunning()

	var progress *templates.ProgressSnapshot
	if running && h.crawler != nil {
		progress = progressSnapshotFrom(h.crawler.Progress())
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerStatusFragment(running, progress).Render(r.Context(), w)
}

// SourceCreate handles POST /admin/crawler/sources. Appends a new entry to
// sources.toml; the next crawl run picks up the change without a restart.
func (h *Handler) SourceCreate(w http.ResponseWriter, r *http.Request) {
	if h.sources == nil {
		http.Error(w, "sources registry not configured", http.StatusServiceUnavailable)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	src := domain.CrawlSource{
		Key:         r.FormValue("key"),
		Name:        r.FormValue("name"),
		SeedURL:     r.FormValue("seed_url"),
		Description: strings.TrimSpace(r.FormValue("description")),
		Tags:        parseTags(r.FormValue("tags")),
		Enabled:     r.FormValue("enabled") == "true",
	}

	current := h.sources.Snapshot()
	for _, existing := range current {
		if existing.Key == src.Key {
			h.sourceError(w, r, "create source", fmt.Errorf("key %q already exists", src.Key))
			return
		}
	}
	current = append(current, src)

	if err := h.sources.Replace(current); err != nil {
		h.sourceError(w, r, "create source", err)
		return
	}
	h.renderCrawler(w, r, "Source created", "success")
}

// SourceUpdate handles PUT /admin/crawler/sources/{key}.
func (h *Handler) SourceUpdate(w http.ResponseWriter, r *http.Request) {
	key := chi.URLParam(r, "key")
	if err := validateSourceKey(key); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if h.sources == nil {
		http.Error(w, "sources registry not configured", http.StatusServiceUnavailable)
		return
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	current := h.sources.Snapshot()
	idx := -1
	for i, s := range current {
		if s.Key == key {
			idx = i
			break
		}
	}
	if idx < 0 {
		http.Error(w, "source not found", http.StatusNotFound)
		return
	}

	current[idx].Name = r.FormValue("name")
	current[idx].SeedURL = r.FormValue("seed_url")
	current[idx].Description = strings.TrimSpace(r.FormValue("description"))
	current[idx].Tags = parseTags(r.FormValue("tags"))
	current[idx].Enabled = r.FormValue("enabled") == "true"

	if err := h.sources.Replace(current); err != nil {
		h.sourceError(w, r, "update source", err)
		return
	}
	h.renderCrawler(w, r, "Source updated", "success")
}

// SourceDelete handles DELETE /admin/crawler/sources/{key}.
func (h *Handler) SourceDelete(w http.ResponseWriter, r *http.Request) {
	key := chi.URLParam(r, "key")
	if err := validateSourceKey(key); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if h.sources == nil {
		http.Error(w, "sources registry not configured", http.StatusServiceUnavailable)
		return
	}

	current := h.sources.Snapshot()
	filtered := make([]domain.CrawlSource, 0, len(current))
	for _, s := range current {
		if s.Key != key {
			filtered = append(filtered, s)
		}
	}
	if len(filtered) == len(current) {
		http.Error(w, "source not found", http.StatusNotFound)
		return
	}

	if err := h.sources.Replace(filtered); err != nil {
		h.sourceError(w, r, "delete source", err)
		return
	}

	// Mark the deleted source's pages stale so the existing Purge Stale flow
	// can drop their embeddings, chunks and stored objects on demand.
	staleCount, markErr := h.store.MarkPagesStaleBySourceKey(r.Context(), key)
	if markErr != nil {
		h.logger.Warn().Err(markErr).Str("key", key).Msg("marking deleted source pages stale")
	}

	msg := "Source deleted"
	if staleCount > 0 {
		msg = fmt.Sprintf("Source deleted; %d pages marked stale, run Purge Stale to remove", staleCount)
	}
	h.renderCrawler(w, r, msg, "success")
}

// sourceError logs and returns a toast for a failed source operation.
func (h *Handler) sourceError(w http.ResponseWriter, r *http.Request, op string, err error) {
	h.logger.Error().Err(err).Str("op", op).Msg("source operation failed")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.Toast(fmt.Sprintf("Failed to %s: %s", op, err.Error()), "error").Render(r.Context(), w)
}

// PurgeStale handles POST /admin/crawler/purge.
func (h *Handler) PurgeStale(w http.ResponseWriter, r *http.Request) {
	count, err := h.store.PurgeStaleDocuments(r.Context())
	if err != nil {
		h.logger.Error().Err(err).Msg("purging stale documents")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.Toast("Purge failed: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	h.renderCrawler(w, r, fmt.Sprintf("Purged %d stale documents", count), "success")
}

// renderCrawler writes the refreshed crawler content fragment plus an optional toast.
// If msg is empty, no toast is rendered.
func (h *Handler) renderCrawler(w http.ResponseWriter, r *http.Request, msg, level string) {
	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	if msg != "" {
		templates.Toast(msg, level).Render(r.Context(), w)
	}
}

// progressSnapshotFrom maps a crawler progress snapshot to its templates-package twin.
func progressSnapshotFrom(snap *crawler.ProgressSnapshot) *templates.ProgressSnapshot {
	if snap == nil {
		return nil
	}
	return &templates.ProgressSnapshot{
		Discovered:  snap.Discovered,
		Fetched:     snap.Fetched,
		CrawlFail:   snap.CrawlFail,
		Parsed:      snap.Parsed,
		ParseFail:   snap.ParseFail,
		Ingested:    snap.Ingested,
		Skipped:     snap.Skipped,
		EmbedFail:   snap.EmbedFail,
		CrawlActive: snap.CrawlActive,
		ParseActive: snap.ParseActive,
		EmbedActive: snap.EmbedActive,
	}
}
