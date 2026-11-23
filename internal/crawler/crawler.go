// Package crawler implements a concurrent BFS web crawler for the UMA website.
package crawler

import (
	"context"
	"crypto/sha256"
	"errors"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

// Crawler orchestrates web crawling of UMA pages.
type Crawler struct {
	store    *store.SQLiteStore
	objStore *objectstore.Client
	ingester *ingest.Ingester
	fetcher  *Fetcher
	browser  *Browser
	config   config.CrawlerConfig
	sources  *sources.Registry
	logger   zerolog.Logger

	mu       sync.RWMutex
	running  bool
	cancel   context.CancelCauseFunc
	progress *RunProgress
}

// New creates a new crawler. The Registry holds the live source snapshot;
// admin edits to sources.toml flow through the same Registry so the next
// crawl run sees them without a server restart.
func New(
	st *store.SQLiteStore,
	obj *objectstore.Client,
	ing *ingest.Ingester,
	cfg config.CrawlerConfig,
	srcs *sources.Registry,
	logger zerolog.Logger,
) *Crawler {
	f := NewFetcher(cfg, logger)

	c := &Crawler{
		store:    st,
		objStore: obj,
		ingester: ing,
		fetcher:  f,
		config:   cfg,
		sources:  srcs,
		logger:   logger,
	}

	// Initialize headless browser if JS patterns are configured
	if len(cfg.JSPatterns) > 0 {
		browser, err := NewBrowser(cfg.RequestTimeout)
		if err != nil {
			logger.Warn().Err(err).Msg("headless browser not available, JS-rendered pages will use static fetch")
		} else {
			c.browser = browser
			f.SetBrowser(browser)
			logger.Info().Strs("patterns", cfg.JSPatterns).Msg("headless browser initialized for JS rendering")
		}
	}

	return c
}

// SetConfig updates the crawler config (thread-safe, takes effect on next crawl).
func (c *Crawler) SetConfig(cfg config.CrawlerConfig) {
	c.mu.Lock()
	c.config = cfg
	c.fetcher.SetConfig(cfg)
	c.mu.Unlock()
}

// snapshotSources returns the current sources for use during a run. The
// Registry returns a defensive copy on every call.
func (c *Crawler) snapshotSources() []domain.CrawlSource {
	if c.sources == nil {
		return nil
	}
	return c.sources.Snapshot()
}

// Close releases browser resources.
func (c *Crawler) Close() {
	if c.browser != nil {
		c.browser.Close()
		c.browser = nil
	}
}

// IsRunning returns whether a crawl is in progress.
func (c *Crawler) IsRunning() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.running
}

// Cancel cancels the current crawl run.
func (c *Crawler) Cancel() {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.cancel != nil {
		c.cancel(nil)
	}
}

// Progress returns a snapshot of the current run's progress, or nil if not running.
func (c *Crawler) Progress() *ProgressSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.progress == nil {
		return nil
	}
	snap := c.progress.Snapshot()
	return &snap
}

// run is the state every stage of one crawl or reprocess run shares.
type run struct {
	id        int64
	startedAt time.Time
	progress  *RunProgress
	cancel    context.CancelCauseFunc // a non-nil cause fails the run instead of cancelling it
}

// withRunLock acquires the run lock, executes fn, and releases on return.
// Returns an error if a run is already in progress.
func (c *Crawler) withRunLock(ctx context.Context, fn func(context.Context, *run) (int64, error)) (int64, error) {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return 0, fmt.Errorf("crawl already running")
	}
	c.running = true
	r := &run{progress: &RunProgress{}}
	ctx, r.cancel = context.WithCancelCause(ctx)
	c.progress = r.progress
	c.cancel = r.cancel
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.running = false
		c.cancel = nil
		c.progress = nil
		c.mu.Unlock()
	}()

	return fn(ctx, r)
}

// RunAll crawls all enabled sources concurrently. Returns the run ID.
func (c *Crawler) RunAll(ctx context.Context) (int64, error) {
	return c.withRunLock(ctx, c.runCrawl)
}

// Reprocess re-runs the parse and embed stages for all fetched pages
// using data from the object store, skipping the crawl stage entirely.
// Useful when parsing/chunking/embedding code changes but crawled data is still fresh.
func (c *Crawler) Reprocess(ctx context.Context) (int64, error) {
	return c.withRunLock(ctx, c.runReprocess)
}

func (c *Crawler) runCrawl(ctx context.Context, r *run) (int64, error) {
	if err := c.begin(ctx, r); err != nil {
		return 0, err
	}

	var enabled []domain.CrawlSource
	for _, s := range c.snapshotSources() {
		if s.Enabled {
			enabled = append(enabled, s)
		}
	}
	if len(enabled) == 0 {
		c.finishRun(r.id, domain.CrawlRunCompleted, ProgressSnapshot{}, nil)
		return r.id, nil
	}

	c.logger.Info().
		Int64("run_id", r.id).
		Int("sources", len(enabled)).
		Int("max_pages", c.config.MaxPages).
		Int("max_depth", c.config.MaxDepth).
		Msg("crawl started")

	c.pipeline(ctx, r, "crawl", false, func(parseCh chan<- CrawlItem) {
		c.crawlStageAll(ctx, r, enabled, parseCh)
	})

	status, runErr := runOutcome(ctx)
	snap := r.progress.Snapshot()
	c.logger.Info().
		Err(runErr).
		Int64("run_id", r.id).
		Str("status", string(status)).
		Int("discovered", int(snap.Discovered)).
		Int("fetched", int(snap.Fetched)).
		Int("ingested", int(snap.Ingested)).
		Int("skipped", int(snap.Skipped)).
		Int("failed", int(snap.Failed())).
		Dur("duration", time.Since(r.startedAt)).
		Msg("crawl finished")

	c.finishRun(r.id, status, snap, runErr)
	return r.id, runErr
}

func (c *Crawler) runReprocess(ctx context.Context, r *run) (int64, error) {
	if err := c.begin(ctx, r); err != nil {
		return 0, err
	}

	all, err := c.store.ListCrawlPages(ctx, nil, nil)
	if err != nil {
		return r.id, fmt.Errorf("listing crawl pages: %w", err)
	}
	// A failed page keeps the document of its last good crawl, so it is
	// replayed too; otherwise that document would keep the old embeddings.
	var pages []domain.CrawlPage
	for _, p := range all {
		if p.Status == domain.CrawlPageFetched || (p.Status == domain.CrawlPageFailed && p.DocID != nil) {
			pages = append(pages, p)
		}
	}
	if len(pages) == 0 {
		c.finishRun(r.id, domain.CrawlRunCompleted, ProgressSnapshot{}, nil)
		return r.id, nil
	}

	c.logger.Info().
		Int64("run_id", r.id).
		Int("pages", len(pages)).
		Msg("reprocess started")

	c.pipeline(ctx, r, "reprocess", true, func(parseCh chan<- CrawlItem) {
		c.replayStage(ctx, pages, parseCh, r.progress)
	})

	// After a model change the embedding cache invalidates itself on the first
	// dim-mismatched insert and stays unloaded until restart, so re-Warm it here.
	if n, err := c.store.Warm(cleanupCtx(ctx)); err != nil {
		c.logger.Warn().Err(err).Int64("run_id", r.id).Msg("post-reprocess cache warm failed; SearchSimilar will use SQL fallback")
	} else {
		c.logger.Info().Int64("run_id", r.id).Int("vectors", n).Msg("post-reprocess cache warmed")
	}

	status, runErr := runOutcome(ctx)
	snap := r.progress.Snapshot()
	c.logger.Info().
		Err(runErr).
		Int64("run_id", r.id).
		Str("status", string(status)).
		Int("pages", int(snap.Discovered)).
		Int("ingested", int(snap.Ingested)).
		Int("failed", int(snap.Failed())).
		Dur("duration", time.Since(r.startedAt)).
		Msg("reprocess finished")

	c.finishRun(r.id, status, snap, runErr)
	return r.id, runErr
}

// begin fails fast on an unreachable embedder or object store, before any run
// row or partial side effect exists, then records the run. Without the bucket
// every raw body and image would be dropped with only a warning per page.
func (c *Crawler) begin(ctx context.Context, r *run) error {
	if err := c.ingester.HealthCheck(ctx); err != nil {
		return fmt.Errorf("health check: %w", err)
	}
	if !c.objStore.Healthy(ctx) {
		return fmt.Errorf("health check: object store bucket unreachable")
	}
	id, err := c.store.CreateCrawlRun(ctx, nil)
	if err != nil {
		return fmt.Errorf("creating crawl run: %w", err)
	}
	r.id = id
	r.startedAt = time.Now()
	return nil
}

// pipeline runs feed into the parse and embed worker pools and returns once
// every stage has drained. force makes the embed stage re-embed documents that
// already exist (reprocess).
func (c *Crawler) pipeline(ctx context.Context, r *run, label string, force bool, feed func(chan<- CrawlItem)) {
	parseCh := make(chan CrawlItem, 16)
	embedCh := make(chan ParseItem, 16)

	stopTicker := c.startProgressTicker(r, label)

	var parseWG, embedWG sync.WaitGroup
	for range c.config.EmbedWorkers {
		embedWG.Go(func() { c.embedStage(ctx, embedCh, r.progress, func() { r.cancel(errEmbedFailures) }, force) })
	}
	for range c.config.ParseWorkers {
		parseWG.Go(func() { c.parseStage(ctx, parseCh, embedCh, r.progress) })
	}

	// Cascade close: feed done → close parseCh → parsers drain → close embedCh → embedders drain.
	feed(parseCh)
	close(parseCh)
	parseWG.Wait()
	close(embedCh)
	embedWG.Wait()
	stopTicker()

	c.purgeOrphans(ctx, r.id)
}

// startProgressTicker logs progress at CrawlerConfig.ProgressLogInterval and returns a stop function.
func (c *Crawler) startProgressTicker(r *run, label string) func() {
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(c.config.ProgressLogInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				snap := r.progress.Snapshot()
				c.logger.Info().
					Int64("run_id", r.id).
					Int64("discovered", snap.Discovered).
					Int64("fetched", snap.Fetched).
					Int64("parsed", snap.Parsed).
					Int64("ingested", snap.Ingested).
					Int64("failed", snap.Failed()).
					Dur("elapsed", time.Since(r.startedAt)).
					Msg(label + " progress")
			case <-done:
				return
			}
		}
	}()
	return func() { close(done) }
}

// errEmbedFailures is the cause recorded when max_embed_fails consecutive
// ingest failures abort a run.
var errEmbedFailures = errors.New("too many consecutive embed failures")

// runOutcome reports how the run ended: on its own, cancelled by the operator
// or shutdown, or failed with the cause that aborted it.
func runOutcome(ctx context.Context) (domain.CrawlRunStatus, error) {
	switch cause := context.Cause(ctx); {
	case cause == nil:
		return domain.CrawlRunCompleted, nil
	case errors.Is(cause, context.Canceled):
		return domain.CrawlRunCancelled, nil
	default:
		return domain.CrawlRunFailed, cause
	}
}

// cleanupCtx returns ctx while it is live and a background context once it is
// cancelled, so end-of-run cleanup still runs after a cancelled crawl.
func cleanupCtx(ctx context.Context) context.Context {
	if ctx.Err() != nil {
		return context.Background()
	}
	return ctx
}

// purgeOrphans removes documents no longer referenced by any crawl_page.
func (c *Crawler) purgeOrphans(ctx context.Context, runID int64) {
	n, err := c.store.PurgeOrphanDocuments(cleanupCtx(ctx))
	if err != nil {
		c.logger.Warn().Err(err).Int64("run_id", runID).Msg("purging orphan documents failed")
		return
	}
	if n > 0 {
		c.logger.Info().Int64("run_id", runID).Int("orphans", n).Msg("purged orphan documents")
	}
}

// objectStoreKey returns the RustFS key for a crawled page's raw body.
func objectStoreKey(contentType, pageURL string) string {
	h := sha256.Sum256([]byte(pageURL))
	return fmt.Sprintf("crawl/%s/%x", contentType, h)
}

// imageObjectStoreKey returns the RustFS key for an image artifact rendered
// from the page at pageURL. Refs are page tags like "p1", "p2", or "full".
func imageObjectStoreKey(pageURL, ref string) string {
	h := sha256.Sum256([]byte(pageURL))
	return fmt.Sprintf("crawl/img/%x/%s.png", h, ref)
}

// titleFromURL extracts a title from a URL (for PDFs which lack HTML title).
func titleFromURL(u string) string {
	parsed, err := url.Parse(u)
	if err != nil {
		return u
	}
	parts := strings.Split(parsed.Path, "/")
	for i := len(parts) - 1; i >= 0; i-- {
		if parts[i] != "" {
			return strings.TrimSuffix(parts[i], ".pdf")
		}
	}
	return u
}

// recordFailedPage marks pageURL failed in this run. Stamping last_crawled_at
// keeps the end-of-source stale sweep from relabelling the failure as stale.
// A page the site answers 404 or 410 for is gone, so its row drops the
// document and the end-of-run orphan purge deletes it; any other failure keeps
// the document of the last good crawl.
func (c *Crawler) recordFailedPage(ctx context.Context, sourceKey, pageURL string, fetchErr error) {
	errMsg := fetchErr.Error()
	now := time.Now()
	page := &domain.CrawlPage{
		SourceKey:     sourceKey,
		URL:           pageURL,
		Status:        domain.CrawlPageFailed,
		ErrorMessage:  &errMsg,
		LastCrawledAt: &now,
	}
	record := c.store.MarkCrawlPageFailed
	if pageGone(fetchErr) {
		record = c.store.UpsertCrawlPage
	}
	if err := record(ctx, page); err != nil {
		c.logger.Warn().Err(err).Str("url", pageURL).Msg("failed to record page failure")
	}
}

// pageGone reports whether err is the site saying the page no longer exists.
func pageGone(err error) bool {
	var status *statusError
	return errors.As(err, &status) && (status.code == http.StatusNotFound || status.code == http.StatusGone)
}

func (c *Crawler) finishRun(runID int64, status domain.CrawlRunStatus, snap ProgressSnapshot, runErr error) {
	now := time.Now()
	var errMsg *string
	if runErr != nil {
		msg := runErr.Error()
		errMsg = &msg
	}
	run := &domain.CrawlRun{
		ID:              runID,
		Status:          status,
		PagesDiscovered: int(snap.Discovered),
		PagesFetched:    int(snap.Fetched),
		PagesChanged:    int(snap.Ingested),
		PagesFailed:     int(snap.Failed()),
		ErrorMessage:    errMsg,
		FinishedAt:      &now,
	}

	// Use background context since the crawl context may be cancelled
	if err := c.store.UpdateCrawlRun(context.Background(), run); err != nil {
		c.logger.Error().Err(err).Int64("run_id", runID).Msg("failed to update crawl run")
	}
}
