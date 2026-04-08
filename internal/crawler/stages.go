package crawler

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
)

// crawlStageAll runs BFS crawl for all enabled sources concurrently, sending items to parseCh.
// It blocks until all sources have finished crawling.
func (c *Crawler) crawlStageAll(
	ctx context.Context,
	sources []domain.CrawlSource,
	parseCh chan<- CrawlItem,
	progress *RunProgress,
	runStartedAt time.Time,
) {
	progress.CrawlActive.Store(1)
	defer progress.CrawlActive.Store(0)

	sem := make(chan struct{}, c.config.Concurrency)
	var globalPages atomic.Int64
	var globalVisited sync.Map
	var seenSubjects sync.Map // subject code → true (dedup across plans)
	var wg sync.WaitGroup

	for _, src := range sources {
		wg.Add(1)
		go func() {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			defer func() { <-sem }()

			if err := c.crawlSourceStage(ctx, src, parseCh, progress, &globalPages, &globalVisited, &seenSubjects); err != nil {
				c.logger.Error().Err(err).Str("source", src.Name).Msg("source crawl failed")
			}

			// Mark stale pages after source completes
			stale, err := c.store.MarkStalePages(ctx, src.ID, runStartedAt)
			if err != nil {
				c.logger.Warn().Err(err).Str("source", src.Name).Msg("failed to mark stale pages")
			} else if stale > 0 {
				c.logger.Info().Int("stale", stale).Str("source", src.Name).Msg("marked stale pages")
			}
		}()
	}

	wg.Wait()
}

// crawlSourceStage runs BFS for a single source, sending CrawlItems to parseCh.
func (c *Crawler) crawlSourceStage(
	ctx context.Context,
	src domain.CrawlSource,
	parseCh chan<- CrawlItem,
	progress *RunProgress,
	globalPages *atomic.Int64,
	globalVisited *sync.Map,
	seenSubjects *sync.Map,
) error {
	c.logger.Info().Str("source", src.Name).Str("seed", src.SeedURL).Msg("starting source crawl")

	type queueItem struct {
		url   string
		depth int
	}

	queue := []queueItem{{url: src.SeedURL, depth: 0}}
	globalVisited.Store(src.SeedURL, true)

	// Per-source counters for completion summary
	var srcDiscovered, srcFetched, srcFailed int
	var maxDepthReached int

	seedParsed, err := url.Parse(src.SeedURL)
	if err != nil {
		return fmt.Errorf("parsing seed URL: %w", err)
	}
	seedPath := seedParsed.Path

	srcRef := CrawlSourceRef{
		ID:       src.ID,
		Category: src.Category,
		Language: src.Language,
	}

	ticker := time.NewTicker(c.fetcher.Delay())
	defer ticker.Stop()

	for len(queue) > 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}

		if globalPages.Load() >= int64(c.config.MaxPages) {
			c.logger.Info().Str("source", src.Name).Msg("global page limit reached")
			break
		}

		item := queue[0]
		queue[0] = queueItem{} // zero out to allow GC of url string
		queue = queue[1:]

		if item.depth > c.config.MaxDepth {
			continue
		}

		globalPages.Add(1)
		progress.Discovered.Add(1)
		srcDiscovered++
		if item.depth > maxDepthReached {
			maxDepthReached = item.depth
		}

		result, err := c.fetcher.Fetch(ctx, item.url)
		if err != nil {
			c.logger.Warn().Err(err).Str("url", item.url).Msg("fetch failed")
			progress.CrawlFail.Add(1)
			srcFailed++
			c.recordFailedPage(ctx, src.ID, item.url, err)
			continue
		}
		progress.Fetched.Add(1)
		srcFetched++

		// Store raw bytes in object store
		objKey := objectStoreKey(result.ContentType, item.url)
		contentTypeHeader := "text/html"
		if result.ContentType == "pdf" {
			contentTypeHeader = "application/pdf"
		}
		if err := c.objStore.Put(ctx, objKey, result.Body, contentTypeHeader); err != nil {
			c.logger.Warn().Err(err).Str("url", item.url).Msg("object store put failed")
		}

		ci := CrawlItem{
			URL:         item.url,
			ContentType: result.ContentType,
			Body:        result.Body,
			ObjKey:      objKey,
			Source:      srcRef,
		}

		// For HTML: parse now (need links for BFS), pass result to parse stage
		if result.ContentType != "pdf" {
			parsed, err := ParseHTML(result.Body, item.url)
			if err != nil {
				c.logger.Warn().Err(err).Str("url", item.url).Msg("html parse failed")
				progress.CrawlFail.Add(1)
				c.recordFailedPage(ctx, src.ID, item.url, err)
				continue
			}
			ci.HTMLParsed = parsed
			// Don't keep raw body for HTML — parse result has everything
			ci.Body = nil

			// Enqueue discovered links
			for _, link := range parsed.Links {
				if _, alreadySeen := globalVisited.LoadOrStore(link, true); alreadySeen {
					continue
				}
				if !shouldFollow(link, seedPath) {
					continue
				}
				// Dedup subject pages: same subject code across different plans
				if code := subjectCode(link); code != "" {
					if _, seen := seenSubjects.LoadOrStore(code, true); seen {
						continue
					}
				}
				queue = append(queue, queueItem{url: link, depth: item.depth + 1})
			}
			// Links are only needed for BFS; nil out to free memory
			parsed.Links = nil
		}

		// Send to parse stage (blocks if channel full = backpressure)
		select {
		case parseCh <- ci:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	c.logger.Info().
		Str("source", src.Name).
		Int("discovered", srcDiscovered).
		Int("fetched", srcFetched).
		Int("failed", srcFailed).
		Int("depth_reached", maxDepthReached).
		Msg("source finished")
	return nil
}

// parseStage reads CrawlItems, extracts text + computes content hash, sends ParseItems.
// Runs as a worker goroutine; exits when inCh is closed.
func (c *Crawler) parseStage(
	ctx context.Context,
	inCh <-chan CrawlItem,
	outCh chan<- ParseItem,
	progress *RunProgress,
) {
	progress.ParseActive.Add(1)
	defer progress.ParseActive.Add(-1)

	for item := range inCh {
		if ctx.Err() != nil {
			return
		}

		var title, text string

		if item.HTMLParsed != nil {
			// HTML: already parsed in crawl stage
			title = item.HTMLParsed.Title
			text = item.HTMLParsed.Text
			if item.HTMLParsed.Description != "" {
				text = item.HTMLParsed.Description + "\n\n" + text
			}
		} else {
			// PDF: extract text with heading detection
			result, err := ExtractPDFText(ctx, item.Body, c.config.PDF)
			if err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("pdf extraction failed")
				progress.ParseFail.Add(1)
				c.recordFailedPage(ctx, item.Source.ID, item.URL, err)
				continue
			}
			text = result.Text
			if result.Title != "" {
				title = result.Title
			} else {
				title = titleFromURL(item.URL)
			}
		}

		if isLowQualityContent(text, c.config.MinContentLen) {
			c.logger.Debug().Str("url", item.URL).Int("len", len(text)).Msg("skipped low-quality content")
			progress.Skipped.Add(1)
			continue
		}

		contentHash := fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
		progress.Parsed.Add(1)

		pi := ParseItem{
			URL:         item.URL,
			Title:       title,
			Text:        text,
			ContentHash: contentHash,
			ContentType: item.ContentType,
			ObjKey:      item.ObjKey,
			Source:      item.Source,
		}

		select {
		case outCh <- pi:
		case <-ctx.Done():
			return
		}
	}
}

// embedStage reads ParseItems, checks for changes, and ingests into the RAG pipeline.
// Runs as a worker goroutine; exits when inCh is closed.
// After 3 consecutive embed failures, it cancels the crawl via cancelFn.
func (c *Crawler) embedStage(
	ctx context.Context,
	inCh <-chan ParseItem,
	progress *RunProgress,
	runStartedAt time.Time,
	cancelFn context.CancelFunc,
	force bool,
) {
	progress.EmbedActive.Add(1)
	defer progress.EmbedActive.Add(-1)

	var embedIngested, embedSkipped, embedFailed int

	for item := range inCh {
		if ctx.Err() != nil {
			return
		}

		now := time.Now()
		existing, _ := c.store.GetCrawlPageByURL(ctx, item.URL)

		if !force && existing != nil && existing.ContentHash == item.ContentHash {
			// Unchanged — update timestamp only
			existing.LastCrawledAt = &now
			existing.Status = domain.CrawlPageFetched
			if err := c.store.UpsertCrawlPage(ctx, existing); err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert unchanged page failed")
			}
			progress.Skipped.Add(1)
			embedSkipped++
			continue
		}

		// New or changed — ingest
		docID := StableDocID(item.URL)
		category := Categorize(item.URL, item.Source.Category)
		language := domain.Language(item.Source.Language)
		urlStr := item.URL

		// Delete old document if exists (re-ingest)
		if err := c.store.DeleteDocument(ctx, docID); err != nil {
			// "not found" is expected for new documents
			if !errors.Is(err, store.ErrNotFound) {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("delete old document failed")
			}
		}

		doc := domain.Document{
			ID:       docID,
			Title:    item.Title,
			Content:  item.Text,
			URL:      &urlStr,
			Language: language,
			Category: category,
		}

		if _, _, err := c.pipeline.Ingest(ctx, []domain.Document{doc}); err != nil {
			c.logger.Warn().Err(err).Str("url", item.URL).Msg("ingest failed")
			progress.EmbedFail.Add(1)
			embedFailed++
			c.recordFailedPage(ctx, item.Source.ID, item.URL, err)
			if progress.ConsecutiveEmbedFails.Add(1) >= int64(c.config.MaxEmbedFails) {
				c.logger.Error().Int("threshold", c.config.MaxEmbedFails).Msg("consecutive embed failures exceeded threshold, aborting crawl")
				cancelFn()
				return
			}
			continue
		}
		progress.ConsecutiveEmbedFails.Store(0)

		progress.Ingested.Add(1)
		embedIngested++
		if embedIngested%25 == 0 {
			c.logger.Info().
				Int("ingested", embedIngested).
				Int("skipped", embedSkipped).
				Int("failed", embedFailed).
				Msg("embed progress")
		}

		page := &domain.CrawlPage{
			SourceID:       item.Source.ID,
			URL:            item.URL,
			ContentType:    item.ContentType,
			Title:          item.Title,
			ContentHash:    item.ContentHash,
			ObjectStoreKey: item.ObjKey,
			DocID:          &docID,
			Status:         domain.CrawlPageFetched,
			LastCrawledAt:  &now,
			LastChangedAt:  &now,
		}
		if err := c.store.UpsertCrawlPage(ctx, page); err != nil {
			c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert page failed")
		}
	}
}
