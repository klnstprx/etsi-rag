// Package crawler implements a concurrent BFS web crawler for the UMA website.
package crawler

import (
	"context"
	"crypto/sha256"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
)

// Crawler orchestrates web crawling of UMA pages.
type Crawler struct {
	store    *store.SQLiteStore
	objStore *objectstore.Client
	pipeline *rag.Pipeline
	fetcher  *Fetcher
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
	return &Crawler{
		store:    st,
		objStore: obj,
		pipeline: pip,
		fetcher:  NewFetcher(cfg.RequestTimeout, cfg.RequestDelay),
		config:   cfg,
		logger:   logger,
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

// RunAll crawls all enabled sources concurrently. Returns the run ID.
func (c *Crawler) RunAll(ctx context.Context) (int64, error) {
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

	return c.runCrawl(ctx, nil)
}

// RunSource crawls a single source. Returns the run ID.
func (c *Crawler) RunSource(ctx context.Context, sourceID int64) (int64, error) {
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

	return c.runCrawl(ctx, &sourceID)
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

	progress := c.progress

	// Pipeline channels
	parseCh := make(chan CrawlItem, 16)
	embedCh := make(chan ParseItem, 16)

	// Start embed workers (2)
	var embedWG sync.WaitGroup
	for range 2 {
		embedWG.Add(1)
		go func() {
			defer embedWG.Done()
			c.embedStage(ctx, embedCh, progress, runStartedAt, cancel)
		}()
	}

	// Start parse workers (4)
	var parseWG sync.WaitGroup
	for range 4 {
		parseWG.Add(1)
		go func() {
			defer parseWG.Done()
			c.parseStage(ctx, parseCh, embedCh, progress)
		}()
	}

	// Run crawl stage (blocks until all sources done)
	c.crawlStageAll(ctx, enabled, parseCh, progress, runStartedAt)

	// Cascade close: crawl done → close parseCh → parsers drain → close embedCh → embedders drain
	close(parseCh)
	parseWG.Wait()
	close(embedCh)
	embedWG.Wait()

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

	c.finishRun(ctx, runID, status, nil, discovered, fetched, changed, failed)
	return runID, nil
}

// shouldFollow decides whether a discovered link should be crawled.
func shouldFollow(link, seedPath string) bool {
	parsed, err := url.Parse(link)
	if err != nil {
		return false
	}

	host := strings.ToLower(parsed.Hostname())
	path := strings.ToLower(parsed.Path)

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
