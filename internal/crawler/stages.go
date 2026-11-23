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
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/markers"
)

// crawlStageAll runs BFS crawl for all enabled sources concurrently, sending items to parseCh.
// It blocks until all sources have finished crawling.
func (c *Crawler) crawlStageAll(ctx context.Context, r *run, sources []domain.CrawlSource, parseCh chan<- CrawlItem) {
	r.progress.CrawlActive.Store(1)
	defer r.progress.CrawlActive.Store(0)

	sem := make(chan struct{}, c.config.Concurrency)
	var f frontier
	// Claim every seed up front so another source's links never queue it:
	// each seed is fetched once per pass, by the source that owns it.
	for _, src := range sources {
		f.visited.Store(normalizeURL(src.SeedURL, c.config.StripQueryParams), true)
	}
	var wg sync.WaitGroup

	for _, src := range sources {
		wg.Go(func() {
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			defer func() { <-sem }()

			if err := c.crawlSourceStage(ctx, src, parseCh, r.progress, &f); err != nil {
				c.logger.Error().Err(err).Str("source", src.Name).Msg("source crawl failed")
			}

			// Mark stale pages after source completes
			stale, err := c.store.MarkStalePages(ctx, src.Key, r.startedAt)
			if err != nil {
				c.logger.Warn().Err(err).Str("source", src.Name).Msg("failed to mark stale pages")
			} else if stale > 0 {
				c.logger.Info().Int("stale", stale).Str("source", src.Name).Msg("marked stale pages")
			}
		})
	}

	wg.Wait()
}

// frontier is the link-admission state shared by every source of one crawl.
type frontier struct {
	pages    atomic.Int64 // pages taken off any queue, capped by MaxPages
	visited  sync.Map     // normalised URL → true
	subjects sync.Map     // subjectKey → true once a page of that plan's subject was kept
}

// admit reports whether a normalised link should be queued: followable from
// seedPath and not seen before. The scope check comes first, so a link outside
// this source's scope stays free for the source that owns it.
func (f *frontier) admit(link, seedPath string) bool {
	if !shouldFollow(link, seedPath) {
		return false
	}
	_, seen := f.visited.LoadOrStore(link, true)
	return !seen
}

// crawlSourceStage runs BFS for a single source, sending CrawlItems to parseCh.
func (c *Crawler) crawlSourceStage(
	ctx context.Context,
	src domain.CrawlSource,
	parseCh chan<- CrawlItem,
	progress *RunProgress,
	f *frontier,
) error {
	c.logger.Info().Str("source", src.Name).Str("seed", src.SeedURL).Msg("starting source crawl")

	type queueItem struct {
		url   string
		depth int
	}

	seedURL := normalizeURL(src.SeedURL, c.config.StripQueryParams)
	queue := []queueItem{{url: seedURL, depth: 0}}

	// Per-source counters for completion summary
	var srcDiscovered, srcFetched, srcFailed int
	var maxDepthReached int

	seedParsed, err := url.Parse(seedURL)
	if err != nil {
		return fmt.Errorf("parsing seed URL: %w", err)
	}
	seedPath := seedParsed.Path

	// Pacing lives in the fetcher, per host and shared by every source.
	for len(queue) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}

		if f.pages.Load() >= int64(c.config.MaxPages) {
			c.logger.Info().Str("source", src.Name).Msg("global page limit reached")
			break
		}

		item := queue[0]
		queue[0] = queueItem{} // zero out to allow GC of url string
		queue = queue[1:]

		if item.depth > c.config.MaxDepth {
			continue
		}
		// One page per plan's subject; claimed only once a page is kept, so a
		// URL that fails or redirects away leaves the subject's other URLs in play.
		if _, kept := f.subjects.Load(subjectKey(item.url)); kept {
			continue
		}

		f.pages.Add(1)
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
			c.recordFailedPage(ctx, src.Key, item.url, err)
			continue
		}
		progress.Fetched.Add(1)
		srcFetched++

		// A redirected page is recorded under its final URL, which is also the
		// base its relative links resolve against.
		pageURL := item.url
		if final := normalizeURL(result.URL, c.config.StripQueryParams); final != item.url {
			if !f.admit(final, seedPath) {
				c.logger.Debug().Str("url", item.url).Str("final_url", final).Msg("redirect target already seen or out of scope, skipped")
				continue
			}
			pageURL = final
		}
		if subject := subjectKey(pageURL); subject != "" {
			f.subjects.Store(subject, true)
		}

		// Store raw bytes in object store
		objKey := objectStoreKey(result.ContentType, pageURL)
		contentTypeHeader := "text/html"
		if result.ContentType == "pdf" {
			contentTypeHeader = "application/pdf"
		}
		if err := c.objStore.Put(ctx, objKey, result.Body, contentTypeHeader); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Msg("object store put failed")
		}

		ci := CrawlItem{
			URL:         pageURL,
			ContentType: result.ContentType,
			Body:        result.Body,
			ObjKey:      objKey,
			SourceKey:   src.Key,
			Screenshot:  result.Screenshot,
		}

		if err := ci.parseHTML(); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Msg("html parse failed")
			progress.CrawlFail.Add(1)
			c.recordFailedPage(ctx, src.Key, pageURL, err)
			continue
		}
		if parsed := ci.HTMLParsed; parsed != nil {
			// Normalize first so tracking-param variants collapse into one visited entry and one fetch.
			for _, rawLink := range parsed.Links {
				link := normalizeURL(rawLink, c.config.StripQueryParams)
				if f.admit(link, seedPath) {
					queue = append(queue, queueItem{url: link, depth: item.depth + 1})
				}
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

// replayStage stands in for the crawl stage during reprocess: it feeds the
// bodies archived in the object store into parseCh instead of fetching.
func (c *Crawler) replayStage(ctx context.Context, pages []domain.CrawlPage, parseCh chan<- CrawlItem, progress *RunProgress) {
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

		ci := CrawlItem{
			URL:         page.URL,
			ContentType: page.ContentType,
			Body:        body,
			ObjKey:      page.ObjectStoreKey,
			SourceKey:   page.SourceKey,
		}
		if err := ci.parseHTML(); err != nil {
			c.logger.Warn().Err(err).Str("url", page.URL).Msg("html parse failed")
			progress.CrawlFail.Add(1)
			continue
		}

		select {
		case parseCh <- ci:
		case <-ctx.Done():
			return
		}
	}
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

		var title, text, pageLang string
		var artifacts []ParsedArtifact

		if item.HTMLParsed != nil {
			// HTML: already parsed in crawl stage
			title = item.HTMLParsed.Title
			text = item.HTMLParsed.Text
			pageLang = item.HTMLParsed.Language

			// Attach the full-page screenshot as a single "full" artifact when
			// the browser captured one. A single [[image:full]] line at the top
			// of the document acts as a page boundary the chunker reads to
			// stamp every emitted chunk with the "full" ref.
			if len(item.Screenshot) > 0 {
				text = markers.Format("full") + "\n" + text
				artifacts = append(artifacts, ParsedArtifact{
					Ref:      "full",
					Kind:     domain.ArtifactHTMLScreenshot,
					MimeType: "image/png",
					Data:     item.Screenshot,
				})
			}
		} else {
			// PDF: extract text with heading detection (markers emitted at
			// page boundaries by pdfTextToMarkdown), then render each page
			// to a PNG. Markers stay even when image rendering fails — at
			// retrieval time, missing artifacts are silently skipped.
			result, err := ExtractPDFText(ctx, item.Body, c.config.PDF)
			if err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("pdf extraction failed")
				progress.ParseFail.Add(1)
				c.recordFailedPage(ctx, item.SourceKey, item.URL, err)
				continue
			}
			text = result.Text
			if result.Title != "" {
				title = result.Title
			} else {
				title = titleFromURL(item.URL)
			}

			if c.config.PDF.RenderDPI > 0 {
				images, err := ExtractPDFImages(ctx, item.Body, c.config.PDF.RenderDPI)
				if err != nil {
					c.logger.Warn().Err(err).Str("url", item.URL).Msg("pdf image render failed (continuing without images)")
				} else {
					for _, img := range images {
						artifacts = append(artifacts, ParsedArtifact{
							Ref:      img.Ref,
							Kind:     domain.ArtifactPDFPage,
							MimeType: "image/png",
							Data:     img.PNG,
						})
					}
				}
			}
		}

		// Judge and hash the text without image markers, so a screenshot that
		// fails on one fetch does not skip the page or change its document.
		body, _ := markers.ExtractBoundaries(text)
		if isLowQualityContent(body, c.config.MinContentLen, c.config.RejectionPatterns) {
			c.logger.Debug().Str("url", item.URL).Int("len", len(body)).Msg("skipped low-quality content")
			progress.Skipped.Add(1)
			continue
		}

		contentHash := fmt.Sprintf("%x", sha256.Sum256([]byte(body)))
		progress.Parsed.Add(1)

		pi := ParseItem{
			URL:         item.URL,
			Title:       title,
			Text:        text,
			ContentHash: contentHash,
			ContentType: item.ContentType,
			Language:    pageLang,
			ObjKey:      item.ObjKey,
			SourceKey:   item.SourceKey,
			Artifacts:   artifacts,
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
// After MaxEmbedFails consecutive embed failures, it cancels the crawl via cancelFn.
func (c *Crawler) embedStage(
	ctx context.Context,
	inCh <-chan ParseItem,
	progress *RunProgress,
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
		existing, _ := c.store.CrawlPageByURL(ctx, item.URL)

		if !force && existing != nil && existing.DocID != nil && existing.ContentHash == item.ContentHash {
			// URL + content unchanged — bump timestamp only, and retry any
			// artifact an earlier crawl failed to store.
			existing.LastCrawledAt = &now
			existing.Status = domain.CrawlPageFetched
			existing.ErrorMessage = nil
			if err := c.store.UpsertCrawlPage(ctx, existing); err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert unchanged page failed")
			}
			c.storeArtifacts(ctx, item.URL, *existing.DocID, item.Artifacts)
			progress.Skipped.Add(1)
			embedSkipped++
			continue
		}

		docID := StableDocID(item.ContentHash)
		docURL, docTitle := item.URL, item.Title
		page := &domain.CrawlPage{
			SourceKey:      item.SourceKey,
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
		if force && existing != nil {
			// Reprocess fetches nothing, so the page keeps the outcome of its
			// last crawl: a page that failed it stays failed.
			page.Status = existing.Status
			page.ErrorMessage = existing.ErrorMessage
			page.LastCrawledAt = existing.LastCrawledAt
		}

		// Content-addressed dedup: the same content reached via another URL (or
		// an earlier crawl) is stored once. force (reprocess) replaces it so the
		// current model re-embeds it instead of hitting the documents PK.
		docExists, err := c.store.DocumentExists(ctx, docID)
		if err != nil {
			c.logger.Warn().Err(err).Str("url", item.URL).Msg("document existence check failed")
		}
		if docExists {
			if !force {
				c.recordDuplicate(ctx, page, docID)
				progress.Skipped.Add(1)
				embedSkipped++
				continue
			}
			docURL, docTitle, err = c.dropForReembed(ctx, docID, item.URL, item.Title)
			if err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Str("doc_id", docID).Msg("delete stale document failed")
				progress.EmbedFail.Add(1)
				embedFailed++
				c.recordFailedPage(ctx, item.SourceKey, item.URL, err)
				continue
			}
		}

		doc := domain.Document{
			ID:       docID,
			Title:    docTitle,
			Content:  item.Text,
			URL:      &docURL,
			Language: pageLanguage(item.Language),
		}

		if _, _, err := c.ingester.Ingest(ctx, []domain.Document{doc}); err != nil {
			// Chunking failures (e.g. tokenizer quirks on malformed sections) are
			// document-specific and shouldn't trip the consecutive-embed-fail circuit
			// breaker — the backend is healthy, the document just can't be processed.
			chunkErr := errors.Is(err, ingest.ErrChunking)
			if chunkErr {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("chunking failed")
			} else {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("ingest failed")
			}
			progress.EmbedFail.Add(1)
			embedFailed++
			c.recordFailedPage(ctx, item.SourceKey, item.URL, err)
			if chunkErr {
				continue
			}
			if progress.ConsecutiveEmbedFails.Add(1) >= int64(c.config.MaxEmbedFails) {
				c.logger.Error().Int("threshold", c.config.MaxEmbedFails).Msg("consecutive embed failures exceeded threshold, aborting crawl")
				cancelFn()
				return
			}
			continue
		}
		progress.ConsecutiveEmbedFails.Store(0)

		// Persist image artifacts AFTER ingest so doc_id exists. Artifact
		// failures don't fail the document — text retrieval still works.
		c.storeArtifacts(ctx, item.URL, docID, item.Artifacts)

		progress.Ingested.Add(1)
		embedIngested++
		if embedIngested%25 == 0 {
			c.logger.Info().
				Int("ingested", embedIngested).
				Int("skipped", embedSkipped).
				Int("failed", embedFailed).
				Msg("embed progress")
		}

		if err := c.store.UpsertCrawlPage(ctx, page); err != nil {
			c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert page failed")
		}
	}
}

// pageLanguage returns the language parsed from <html lang>, falling back to
// Spanish because the corpus is UMA-only Spanish content.
func pageLanguage(lang string) domain.Language {
	if lang == "" {
		return domain.LanguageES
	}
	return domain.Language(lang)
}

// recordDuplicate links page to the document already stored for its content.
// UMA's CMS serves identical content at a bare program URL and its /cms/menu/...
// sub-pages, so documents.url and its title move to the more canonical page
// instead of keeping whichever page won the embed race.
func (c *Crawler) recordDuplicate(ctx context.Context, page *domain.CrawlPage, docID string) {
	if err := c.store.UpsertCrawlPage(ctx, page); err != nil {
		c.logger.Warn().Err(err).Str("url", page.URL).Msg("upsert page failed")
	}
	existingURL, err := c.store.DocumentURL(ctx, docID)
	if err != nil || !preferCanonical(page.URL, existingURL) {
		return
	}
	if err := c.store.UpdateDocumentPage(ctx, docID, page.URL, page.Title); err != nil {
		c.logger.Warn().Err(err).Str("doc_id", docID).Str("url", page.URL).Msg("canonical url swap failed")
		return
	}
	c.logger.Debug().Str("doc_id", docID).Str("from", existingURL).Str("to", page.URL).Msg("documents.url swapped to more canonical")
}

// dropForReembed deletes docID (chunks and embeddings cascade) so reprocess can
// re-ingest it, and returns the URL and title the new row should carry. Pages
// sharing the content collapse to one docID, so the more canonical of the
// stored page and pageURL wins rather than whichever page is reprocessed last;
// reprocessing the stored page itself refreshes its title.
func (c *Crawler) dropForReembed(ctx context.Context, docID, pageURL, pageTitle string) (docURL, docTitle string, err error) {
	docURL, docTitle = pageURL, pageTitle
	if existing, err := c.store.Document(ctx, docID); err == nil && existing.URL != nil &&
		*existing.URL != pageURL && !preferCanonical(pageURL, *existing.URL) {
		docURL, docTitle = *existing.URL, existing.Title
	}
	if err := c.store.DeleteDocument(ctx, docID); err != nil {
		return "", "", err
	}
	return docURL, docTitle, nil
}

// storeArtifacts uploads each rendered image docID has no row for yet to
// RustFS and records it in document_artifacts, so an image that failed on an
// earlier crawl is retried. Failures are logged but never propagate —
// text-only retrieval remains correct without images.
func (c *Crawler) storeArtifacts(ctx context.Context, pageURL, docID string, arts []ParsedArtifact) {
	for _, art := range arts {
		if _, err := c.store.Artifact(ctx, docID, art.Ref); err == nil {
			continue
		}
		key := imageObjectStoreKey(pageURL, art.Ref)
		if err := c.objStore.Put(ctx, key, art.Data, art.MimeType); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Str("ref", art.Ref).Msg("artifact put failed")
			continue
		}
		row := &domain.Artifact{
			DocID:          docID,
			Ref:            art.Ref,
			Kind:           art.Kind,
			ObjectStoreKey: key,
			MimeType:       art.MimeType,
		}
		if err := c.store.UpsertArtifact(ctx, row); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Str("ref", art.Ref).Msg("artifact upsert failed")
		}
	}
}

// StableDocID generates a content-addressed document ID from the SHA-256
// content hash hex string. Two URLs with identical extracted text map to the
// same ID, so duplicate content is stored once and shared across crawl_pages.
func StableDocID(contentHash string) string {
	return "crawl-" + contentHash[:min(len(contentHash), 24)]
}
