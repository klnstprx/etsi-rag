package admin

import (
	"context"
	"fmt"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/domain"
)

func (h *Handler) crawlerData(ctx context.Context) templates.CrawlerData {
	running := h.crawler != nil && h.crawler.IsRunning()

	docCount, _ := h.store.CountDocuments(ctx)
	sources, _ := h.store.ListCrawlSources(ctx)
	runs, _ := h.store.ListCrawlRuns(ctx, 10)

	var progress *templates.ProgressSnapshot
	if running && h.crawler != nil {
		if snap := h.crawler.Progress(); snap != nil {
			progress = &templates.ProgressSnapshot{
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
		data := h.crawlerData(r.Context())
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.CrawlerContent(data).Render(r.Context(), w)
		templates.Toast(msg, level).Render(r.Context(), w)
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
		runID, err := h.crawler.RunAll(context.Background())
		if err != nil {
			h.logger.Error().Err(err).Int64("run_id", runID).Msg("crawl failed")
		} else {
			h.logger.Info().Int64("run_id", runID).Msg("crawl completed")
		}
	}()

	h.logger.Info().Msg("crawler started")

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Crawl started", "success").Render(r.Context(), w)
}

// CrawlerReprocess handles POST /admin/crawler/reprocess.
func (h *Handler) CrawlerReprocess(w http.ResponseWriter, r *http.Request) {
	if h.crawlerGuard(w, r, "reprocess") {
		return
	}

	go func() {
		runID, err := h.crawler.Reprocess(context.Background())
		if err != nil {
			h.logger.Error().Err(err).Int64("run_id", runID).Msg("reprocess failed")
		} else {
			h.logger.Info().Int64("run_id", runID).Msg("reprocess completed")
		}
	}()

	h.logger.Info().Msg("reprocess started")

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Reprocess started", "success").Render(r.Context(), w)
}

// CrawlerStop handles DELETE /admin/crawler/run.
func (h *Handler) CrawlerStop(w http.ResponseWriter, r *http.Request) {
	if h.crawler != nil {
		h.crawler.Cancel()
	}

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Crawl stop requested", "success").Render(r.Context(), w)
}

// CrawlerStatus handles GET /admin/crawler/status (polled by HTMX).
func (h *Handler) CrawlerStatus(w http.ResponseWriter, r *http.Request) {
	running := h.crawler != nil && h.crawler.IsRunning()

	var progress *templates.ProgressSnapshot
	if running && h.crawler != nil {
		if snap := h.crawler.Progress(); snap != nil {
			progress = &templates.ProgressSnapshot{
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
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerStatusFragment(running, progress).Render(r.Context(), w)
}

// SourceCreate handles POST /admin/crawler/sources.
func (h *Handler) SourceCreate(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	src := &domain.CrawlSource{
		Name:     r.FormValue("name"),
		SeedURL:  r.FormValue("seed_url"),
		Category: r.FormValue("category"),
		Language: r.FormValue("language"),
		Enabled:  r.FormValue("enabled") == "true",
	}
	if src.Language == "" {
		src.Language = "es"
	}

	if err := h.store.CreateCrawlSource(r.Context(), src); err != nil {
		h.logger.Error().Err(err).Msg("creating source")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.Toast("Failed to create source: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Source created", "success").Render(r.Context(), w)
}

// SourceUpdate handles PUT /admin/crawler/sources/{id}.
func (h *Handler) SourceUpdate(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(chi.URLParam(r, "id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	src := &domain.CrawlSource{
		ID:       id,
		Name:     r.FormValue("name"),
		SeedURL:  r.FormValue("seed_url"),
		Category: r.FormValue("category"),
		Language: r.FormValue("language"),
		Enabled:  r.FormValue("enabled") == "true",
	}

	if err := h.store.UpdateCrawlSource(r.Context(), src); err != nil {
		h.logger.Error().Err(err).Msg("updating source")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.Toast("Failed to update source: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Source updated", "success").Render(r.Context(), w)
}

// SourceDelete handles DELETE /admin/crawler/sources/{id}.
func (h *Handler) SourceDelete(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(chi.URLParam(r, "id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}

	if err := h.store.DeleteCrawlSource(r.Context(), id); err != nil {
		h.logger.Error().Err(err).Msg("deleting source")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.Toast("Failed to delete source: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast("Source deleted", "success").Render(r.Context(), w)
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

	data := h.crawlerData(r.Context())
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.CrawlerContent(data).Render(r.Context(), w)
	templates.Toast(fmt.Sprintf("Purged %d stale documents", count), "success").Render(r.Context(), w)
}
