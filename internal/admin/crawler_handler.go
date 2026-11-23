package admin

import (
	"context"
	"fmt"
	"net/http"
	"slices"
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
	docCount, _ := h.store.CountDocuments(ctx)
	runs, _ := h.store.ListCrawlRuns(ctx, 10)
	running, progress := h.crawlProgress()
	return templates.CrawlerData{
		Running:  running,
		DocCount: docCount,
		Sources:  h.snapshotSources(),
		Runs:     runs,
		Progress: progress,
	}
}

// crawlProgress reports whether a crawl is running and, if so, its progress.
func (h *Handler) crawlProgress() (bool, *templates.ProgressSnapshot) {
	if h.crawler == nil || !h.crawler.IsRunning() {
		return false, nil
	}
	return true, progressSnapshotFrom(h.crawler.Progress())
}

// CrawlerPage handles GET /admin/crawler.
func (h *Handler) CrawlerPage(w http.ResponseWriter, r *http.Request) {
	data := h.crawlerData(r.Context())
	page := templates.CrawlerPage(data)
	content := templates.CrawlerContent(data)
	render(w, r, page, content)
}

// crawlerGuard checks common preconditions for crawler actions (configured, not running,
// ingester available and reachable). Returns true if the guard failed and a response was sent.
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
	case h.ingester == nil:
		msg, level = "Cannot "+label+": ingester not configured", "error"
	default:
		if err := h.ingester.HealthCheck(r.Context()); err != nil {
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
	h.startRun(w, r, "crawl", "Crawl started", h.crawler.RunAll)
}

// CrawlerReprocess handles POST /admin/crawler/reprocess.
func (h *Handler) CrawlerReprocess(w http.ResponseWriter, r *http.Request) {
	if h.crawlerGuard(w, r, "reprocess") {
		return
	}
	h.startRun(w, r, "reprocess", "Reprocess started", h.crawler.Reprocess)
}

// startRun launches run on the server-lifetime context, since it outlives
// the request, and replies with the refreshed crawler panel.
func (h *Handler) startRun(w http.ResponseWriter, r *http.Request, name, toast string, run func(context.Context) (int64, error)) {
	go func() {
		runID, err := run(h.baseCtx)
		if err != nil {
			h.logger.Error().Err(err).Int64("run_id", runID).Msg(name + " failed")
		} else {
			h.logger.Info().Int64("run_id", runID).Msg(name + " completed")
		}
	}()

	h.logger.Info().Msg(name + " started")
	h.renderCrawler(w, r, toast, "success")
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
	writeHTML(w, r, templates.CrawlerStatusFragment(h.crawlProgress()))
}

// SourceCreate handles POST /admin/crawler/sources. Appends a new entry to
// sources.toml; the next crawl run picks up the change without a restart.
func (h *Handler) SourceCreate(w http.ResponseWriter, r *http.Request) {
	if !h.sourcesReady(w) {
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	src := sourceFromForm(r, r.FormValue("key"))
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
	key, ok := h.sourceKey(w, r)
	if !ok {
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	current := h.sources.Snapshot()
	idx := slices.IndexFunc(current, func(s domain.CrawlSource) bool { return s.Key == key })
	if idx < 0 {
		http.Error(w, "source not found", http.StatusNotFound)
		return
	}
	current[idx] = sourceFromForm(r, key)

	if err := h.sources.Replace(current); err != nil {
		h.sourceError(w, r, "update source", err)
		return
	}
	h.renderCrawler(w, r, "Source updated", "success")
}

// SourceDelete handles DELETE /admin/crawler/sources/{key}.
func (h *Handler) SourceDelete(w http.ResponseWriter, r *http.Request) {
	key, ok := h.sourceKey(w, r)
	if !ok {
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

// sourceKey returns the validated {key} URL parameter, replying 400 or 503
// and returning false when the key is invalid or no registry is wired.
func (h *Handler) sourceKey(w http.ResponseWriter, r *http.Request) (string, bool) {
	key := chi.URLParam(r, "key")
	if err := validateSourceKey(key); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return "", false
	}
	return key, h.sourcesReady(w)
}

// sourcesReady replies 503 and returns false when no registry is wired.
func (h *Handler) sourcesReady(w http.ResponseWriter) bool {
	if h.sources == nil {
		http.Error(w, "sources registry not configured", http.StatusServiceUnavailable)
		return false
	}
	return true
}

// sourceFromForm builds the source with the given key from the create/edit form.
func sourceFromForm(r *http.Request, key string) domain.CrawlSource {
	return domain.CrawlSource{
		Key:         key,
		Name:        r.FormValue("name"),
		SeedURL:     r.FormValue("seed_url"),
		Description: strings.TrimSpace(r.FormValue("description")),
		Tags:        parseTags(r.FormValue("tags")),
		Enabled:     r.FormValue("enabled") == "true",
	}
}

// sourceError logs and returns a toast for a failed source operation.
func (h *Handler) sourceError(w http.ResponseWriter, r *http.Request, op string, err error) {
	h.logger.Error().Err(err).Str("op", op).Msg("source operation failed")
	writeToast(w, r, fmt.Sprintf("Failed to %s: %s", op, err.Error()), "error")
}

// PurgeStale handles POST /admin/crawler/purge.
func (h *Handler) PurgeStale(w http.ResponseWriter, r *http.Request) {
	count, err := h.store.PurgeStaleDocuments(r.Context())
	if err != nil {
		h.logger.Error().Err(err).Msg("purging stale documents")
		writeToast(w, r, "Purge failed: "+err.Error(), "error")
		return
	}

	h.renderCrawler(w, r, fmt.Sprintf("Purged %d stale documents", count), "success")
}

// renderCrawler writes the refreshed crawler content fragment plus a toast.
func (h *Handler) renderCrawler(w http.ResponseWriter, r *http.Request, msg, level string) {
	writeHTML(w, r, templates.CrawlerContent(h.crawlerData(r.Context())), templates.Toast(msg, level))
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
