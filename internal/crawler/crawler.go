// Package crawler implements a concurrent BFS web crawler for the UMA website.
package crawler

import (
	"context"
	"crypto/sha256"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
	"fmt"
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
	pipeline *rag.Pipeline
	fetcher  *Fetcher
	browser  *Browser
	config   config.CrawlerConfig
	logger   zerolog.Logger

	mu       sync.RWMutex
	running  bool
	cancel   context.CancelFunc
	progress *RunProgress
}

// New creates a new crawler.
func New(
	st *store.SQLiteStore,
	obj *objectstore.Client,
	pip *rag.Pipeline,
	cfg config.CrawlerConfig,
	logger zerolog.Logger,
) *Crawler {
	f := NewFetcher(cfg.RequestTimeout, cfg.RequestDelay, cfg.MaxBodySize)
	f.SetJSPatterns(cfg.JSPatterns)

	c := &Crawler{
		store:    st,
		objStore: obj,
		pipeline: pip,
		fetcher:  f,
		config:   cfg,
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
		c.cancel()
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

// withRunLock acquires the run lock, executes fn, and releases on return.
// Returns an error if a run is already in progress.
func (c *Crawler) withRunLock(ctx context.Context, fn func(ctx context.Context) (int64, error)) (int64, error) {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return 0, fmt.Errorf("crawl already running")
	}
	c.running = true
	c.progress = &RunProgress{}
	ctx, c.cancel = context.WithCancel(ctx)
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.running = false
		c.cancel = nil
		c.progress = nil
		c.mu.Unlock()
	}()

	return fn(ctx)
}

// RunAll crawls all enabled sources concurrently. Returns the run ID.
func (c *Crawler) RunAll(ctx context.Context) (int64, error) {
	return c.withRunLock(ctx, func(ctx context.Context) (int64, error) {
		return c.runCrawl(ctx, nil)
	})
}

// RunSource crawls a single source. Returns the run ID.
func (c *Crawler) RunSource(ctx context.Context, sourceID int64) (int64, error) {
	return c.withRunLock(ctx, func(ctx context.Context) (int64, error) {
		return c.runCrawl(ctx, &sourceID)
	})
}

// Reprocess re-runs the parse and embed stages for all fetched pages
// using data from the object store, skipping the crawl stage entirely.
// Useful when parsing/chunking/embedding code changes but crawled data is still fresh.
func (c *Crawler) Reprocess(ctx context.Context) (int64, error) {
	return c.withRunLock(ctx, func(ctx context.Context) (int64, error) {
		return c.runReprocess(ctx)
	})
}

// startProgressTicker logs progress every 30s and returns a stop function.
func (c *Crawler) startProgressTicker(runID int64, progress *RunProgress, startedAt time.Time, label string) func() {
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				snap := progress.Snapshot()
				c.logger.Info().
					Int64("run_id", runID).
					Int64("discovered", snap.Discovered).
					Int64("fetched", snap.Fetched).
					Int64("parsed", snap.Parsed).
					Int64("ingested", snap.Ingested).
					Int64("failed", snap.CrawlFail+snap.ParseFail+snap.EmbedFail).
					Dur("elapsed", time.Since(startedAt)).
					Msg(label + " progress")
			case <-done:
				return
			}
		}
	}()
	return func() { close(done) }
}

// startWorkers launches parse and embed worker goroutines and returns a
// drain function that waits for all workers to complete.
func (c *Crawler) startWorkers(
	ctx context.Context,
	parseCh chan CrawlItem,
	embedCh chan ParseItem,
	progress *RunProgress,
	cancel context.CancelFunc,
	deleteOld bool,
	startedAt time.Time,
) func() {
	var embedWG sync.WaitGroup
	for range c.config.EmbedWorkers {
		embedWG.Add(1)
		go func() {
			defer embedWG.Done()
			c.embedStage(ctx, embedCh, progress, startedAt, cancel, deleteOld)
		}()
	}

	var parseWG sync.WaitGroup
	for range 4 {
		parseWG.Add(1)
		go func() {
			defer parseWG.Done()
			c.parseStage(ctx, parseCh, embedCh, progress)
		}()
	}

	return func() {
		parseWG.Wait()
		close(embedCh)
		embedWG.Wait()
	}
}

func (c *Crawler) runReprocess(ctx context.Context) (int64, error) {
	if err := c.pipeline.Preflight(ctx); err != nil {
		return 0, fmt.Errorf("preflight: %w", err)
	}

	c.mu.RLock()
	cancel := c.cancel
	c.mu.RUnlock()

	runID, err := c.store.CreateCrawlRun(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("creating crawl run: %w", err)
	}
	runStartedAt := time.Now()

	fetched := string(domain.CrawlPageFetched)
	pages, err := c.store.ListCrawlPages(ctx, nil, &fetched)
	if err != nil {
		return runID, fmt.Errorf("listing crawl pages: %w", err)
	}

	// Build source lookup for category/language
	sources, err := c.store.ListCrawlSources(ctx)
	if err != nil {
		return runID, fmt.Errorf("listing sources: %w", err)
	}
	sourceMap := make(map[int64]domain.CrawlSource, len(sources))
	for _, s := range sources {
		sourceMap[s.ID] = s
	}

	if len(pages) == 0 {
		c.finishRun(ctx, runID, domain.CrawlRunCompleted, nil, 0, 0, 0, 0)
		return runID, nil
	}

	c.logger.Info().
		Int64("run_id", runID).
		Int("pages", len(pages)).
		Msg("reprocess started")

	progress := c.progress

	parseCh := make(chan CrawlItem, 16)
	embedCh := make(chan ParseItem, 16)

	stopTicker := c.startProgressTicker(runID, progress, runStartedAt, "reprocess")
	drainWorkers := c.startWorkers(ctx, parseCh, embedCh, progress, cancel, true, runStartedAt)

	// Feed pages from object store into parse channel
	go func() {
		defer close(parseCh)
		for _, page := range pages {
			if ctx.Err() != nil {
				return
			}

			body, err := c.objStore.Get(ctx, page.ObjectStoreKey)
			if err != nil {
				c.logger.Warn().Err(err).Str("url", page.URL).Msg("object store get failed")
				progress.CrawlFail.Add(1)
				continue
			}
			progress.Discovered.Add(1)
			progress.Fetched.Add(1)

			src := sourceMap[page.SourceID]
			ci := CrawlItem{
				URL:         page.URL,
				ContentType: page.ContentType,
				Body:        body,
				ObjKey:      page.ObjectStoreKey,
				Source: CrawlSourceRef{
					ID:       page.SourceID,
					Category: src.Category,
					Language: src.Language,
				},
			}

			// For HTML, parse inline (same as crawl stage but without link extraction)
			if page.ContentType != "pdf" {
				parsed, err := ParseHTML(body, page.URL)
				if err != nil {
					c.logger.Warn().Err(err).Str("url", page.URL).Msg("html parse failed")
					progress.CrawlFail.Add(1)
					continue
				}
				ci.HTMLParsed = parsed
				ci.Body = nil
			}

			select {
			case parseCh <- ci:
			case <-ctx.Done():
				return
			}
		}
	}()

	drainWorkers()
	stopTicker()

	status := domain.CrawlRunCompleted
	if ctx.Err() != nil {
		status = domain.CrawlRunCancelled
	}

	snap := progress.Snapshot()
	discovered := int(snap.Discovered)
	fetchedCount := int(snap.Fetched)
	changed := int(snap.Ingested)
	failed := int(snap.CrawlFail + snap.ParseFail + snap.EmbedFail)

	c.logger.Info().
		Int64("run_id", runID).
		Str("status", string(status)).
		Int("pages", discovered).
		Int("ingested", changed).
		Int("failed", failed).
		Dur("duration", time.Since(runStartedAt)).
		Msg("reprocess finished")

	c.finishRun(ctx, runID, status, nil, discovered, fetchedCount, changed, failed)
	return runID, nil
}

func (c *Crawler) runCrawl(ctx context.Context, sourceID *int64) (int64, error) {
	if err := c.pipeline.Preflight(ctx); err != nil {
		return 0, fmt.Errorf("preflight: %w", err)
	}

	// Capture cancel func for embed abort (set by RunAll/RunSource before this call)
	c.mu.RLock()
	cancel := c.cancel
	c.mu.RUnlock()

	runID, err := c.store.CreateCrawlRun(ctx, sourceID)
	if err != nil {
		return 0, fmt.Errorf("creating crawl run: %w", err)
	}
	runStartedAt := time.Now()

	sources, err := c.store.ListCrawlSources(ctx)
	if err != nil {
		return runID, fmt.Errorf("listing sources: %w", err)
	}

	var enabled []domain.CrawlSource
	for _, s := range sources {
		if !s.Enabled {
			continue
		}
		if sourceID != nil && s.ID != *sourceID {
			continue
		}
		enabled = append(enabled, s)
	}

	if len(enabled) == 0 {
		c.finishRun(ctx, runID, domain.CrawlRunCompleted, nil, 0, 0, 0, 0)
		return runID, nil
	}

	c.logger.Info().
		Int64("run_id", runID).
		Int("sources", len(enabled)).
		Int("max_pages", c.config.MaxPages).
		Int("max_depth", c.config.MaxDepth).
		Msg("crawl started")

	progress := c.progress

	// Pipeline channels
	parseCh := make(chan CrawlItem, 16)
	embedCh := make(chan ParseItem, 16)

	stopTicker := c.startProgressTicker(runID, progress, runStartedAt, "crawl")
	drainWorkers := c.startWorkers(ctx, parseCh, embedCh, progress, cancel, false, runStartedAt)

	// Run crawl stage (blocks until all sources done)
	c.crawlStageAll(ctx, enabled, parseCh, progress, runStartedAt)

	// Cascade close: crawl done → close parseCh → parsers drain → close embedCh → embedders drain
	close(parseCh)
	drainWorkers()
	stopTicker()

	// Determine final status from progress counters
	status := domain.CrawlRunCompleted
	if ctx.Err() != nil {
		status = domain.CrawlRunCancelled
	}

	snap := progress.Snapshot()
	discovered := int(snap.Discovered)
	fetched := int(snap.Fetched)
	changed := int(snap.Ingested)
	failed := int(snap.CrawlFail + snap.ParseFail + snap.EmbedFail)

	c.logger.Info().
		Int64("run_id", runID).
		Str("status", string(status)).
		Int("discovered", discovered).
		Int("fetched", fetched).
		Int("ingested", changed).
		Int("skipped", int(snap.Skipped)).
		Int("failed", failed).
		Dur("duration", time.Since(runStartedAt)).
		Msg("crawl finished")

	c.finishRun(ctx, runID, status, nil, discovered, fetched, changed, failed)
	return runID, nil
}

// blockedExtensions contains file extensions that should never be crawled.
// These are binary or non-text formats that can't be meaningfully embedded.
var blockedExtensions = map[string]bool{
	// Office documents (non-PDF)
	".docx": true, ".dotx": true, ".doc": true,
	".pptx": true, ".ppt": true,
	".xlsx": true, ".xls": true,
	// Images
	".jpg": true, ".jpeg": true, ".png": true, ".gif": true,
	".svg": true, ".webp": true, ".ico": true, ".bmp": true, ".tiff": true,
	// Archives
	".zip": true, ".rar": true, ".gz": true, ".tar": true, ".7z": true,
	// Media
	".mp3": true, ".mp4": true, ".avi": true, ".mov": true, ".wmv": true,
	".wav": true, ".flac": true, ".ogg": true,
	// Executables
	".exe": true, ".msi": true, ".dmg": true, ".apk": true,
}

// hasBlockedExtension returns true if the URL path ends with a blocked file extension.
func hasBlockedExtension(path string) bool {
	lower := strings.ToLower(path)
	// Find last dot in the path (ignore query string)
	if idx := strings.LastIndex(lower, "."); idx >= 0 {
		ext := lower[idx:]
		return blockedExtensions[ext]
	}
	return false
}

// shouldFollow decides whether a discovered link should be crawled.
func shouldFollow(link, seedPath string) bool {
	parsed, err := url.Parse(link)
	if err != nil {
		return false
	}

	host := strings.ToLower(parsed.Hostname())
	path := strings.ToLower(parsed.Path)

	if hasBlockedExtension(path) {
		return false
	}

	switch host {
	case "www.uma.es":
		// Follow: seed path prefix, etsi-informatica, subject pages, media files
		if strings.HasPrefix(path, seedPath) {
			// Skip news
			if strings.Contains(path, "noticias") || strings.Contains(path, "news") || strings.Contains(path, "newsitem") {
				return false
			}
			return true
		}
		if strings.HasPrefix(path, "/etsi-informatica/") ||
			strings.HasPrefix(path, "/centers/subject/") ||
			strings.HasPrefix(path, "/media/files/") {
			return true
		}
		return false

	case "sara.uma.es":
		// Only follow guía docente PDFs
		return strings.Contains(path, "programasasignaturas")

	default:
		return false
	}
}

// objectStoreKey returns the RustFS key for a crawled page.
func objectStoreKey(contentType, pageURL string) string {
	h := sha256.Sum256([]byte(pageURL))
	return fmt.Sprintf("crawl/%s/%x", contentType, h)
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

func (c *Crawler) recordFailedPage(ctx context.Context, sourceID int64, pageURL string, fetchErr error) {
	errMsg := fetchErr.Error()
	page := &domain.CrawlPage{
		SourceID:     sourceID,
		URL:          pageURL,
		Status:       domain.CrawlPageFailed,
		ErrorMessage: &errMsg,
	}
	c.store.UpsertCrawlPage(ctx, page)
}

func (c *Crawler) finishRun(
	ctx context.Context,
	runID int64,
	status domain.CrawlRunStatus,
	runErr error,
	discovered, fetched, changed, failed int,
) {
	now := time.Now()
	run := &domain.CrawlRun{
		ID:              runID,
		Status:          status,
		PagesDiscovered: discovered,
		PagesFetched:    fetched,
		PagesChanged:    changed,
		PagesFailed:     failed,
		FinishedAt:      &now,
	}
	if runErr != nil {
		msg := runErr.Error()
		run.ErrorMessage = &msg
	}

	// Use background context since the crawl context may be cancelled
	if err := c.store.UpdateCrawlRun(context.Background(), run); err != nil {
		c.logger.Error().Err(err).Int64("run_id", runID).Msg("failed to update crawl run")
	}
}
