package crawler

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/rag"
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

	seedURL := normalizeURL(src.SeedURL, c.config.StripQueryParams)
	queue := []queueItem{{url: seedURL, depth: 0}}
	globalVisited.Store(seedURL, true)

	// Per-source counters for completion summary
	var srcDiscovered, srcFetched, srcFailed int
	var maxDepthReached int

	seedParsed, err := url.Parse(seedURL)
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
			Screenshot:  result.Screenshot,
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

			// Enqueue discovered links. Normalize first so tracking-param variants
			// collapse into a single globalVisited entry and one fetch.
			for _, rawLink := range parsed.Links {
				link := normalizeURL(rawLink, c.config.StripQueryParams)
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

		var title, text, pageLang string
		var artifacts []ParsedArtifact

		if item.HTMLParsed != nil {
			// HTML: already parsed in crawl stage
			title = item.HTMLParsed.Title
			text = item.HTMLParsed.Text
			pageLang = item.HTMLParsed.Language

			// Attach the full-page screenshot as a single "full" artifact when
			// the browser captured one. The marker [[image:full]] is prefixed
			// onto every non-heading content line so retrieval reliably finds
			// at least one marker per chunk regardless of how the chunker
			// splits the page.
			if len(item.Screenshot) > 0 {
				text = prefixContentLinesWithMarker(text, markers.Format("full"))
				artifacts = append(artifacts, ParsedArtifact{
					Ref:      "full",
					Kind:     string(domain.ArtifactHTMLScreenshot),
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
				c.recordFailedPage(ctx, item.Source.ID, item.URL, err)
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
					for i, png := range images {
						artifacts = append(artifacts, ParsedArtifact{
							Ref:      fmt.Sprintf("p%d", i+1),
							Kind:     string(domain.ArtifactPDFPage),
							MimeType: "image/png",
							Data:     png,
						})
					}
				}
			}
		}

		if isLowQualityContent(text, c.config.MinContentLen, c.config.RejectionPatterns) {
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
			Language:    pageLang,
			ObjKey:      item.ObjKey,
			Source:      item.Source,
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
			// URL + content unchanged — bump timestamp only.
			existing.LastCrawledAt = &now
			existing.Status = domain.CrawlPageFetched
			if err := c.store.UpsertCrawlPage(ctx, existing); err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert unchanged page failed")
			}
			progress.Skipped.Add(1)
			embedSkipped++
			continue
		}

		docID := StableDocID(item.ContentHash)
		category := Categorize(item.URL, item.Source.Category)
		// Prefer the per-page <html lang> when the parser extracted a supported
		// language; fall back to the source's configured language otherwise
		// (PDFs, pages without lang attr, pages tagged with unsupported langs).
		language := domain.Language(item.Source.Language)
		if item.Language != "" {
			language = domain.Language(item.Language)
		}
		urlStr := item.URL

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

		// Content-addressed dedup: if a document with this ID already exists
		// (same content reached via another URL, or an earlier crawl of this
		// URL), skip the embed and just record the URL — unless force is set,
		// in which case delete the stale doc (chunks/embeddings cascade) so
		// pipeline.Ingest can re-embed with the current model. This is the
		// reprocess-after-model-change path: without the delete, CreateDocument
		// would hit the PK and stale embeddings would linger.
		docExists, err := c.store.DocumentExists(ctx, docID)
		if err != nil {
			c.logger.Warn().Err(err).Str("url", item.URL).Msg("document existence check failed")
		}
		if docExists {
			if !force {
				if err := c.store.UpsertCrawlPage(ctx, page); err != nil {
					c.logger.Warn().Err(err).Str("url", item.URL).Msg("upsert page failed")
				}
				// UMA's CMS serves identical content at the bare program URL and at
				// one of its /cms/menu/... sub-pages. Whoever hits the embed worker
				// first stamps documents.url, so on re-crawl the canonical bare URL
				// can lose the race to a sub-page and disappear from eval/lookup.
				// Swap to the more canonical URL when this happens — see
				// preferCanonical in crawler.go.
				if existingURL, err := c.store.GetDocumentURL(ctx, docID); err == nil {
					if preferCanonical(item.URL, existingURL) {
						if err := c.store.UpdateDocumentURL(ctx, docID, item.URL); err != nil {
							c.logger.Warn().Err(err).Str("doc_id", docID).Str("url", item.URL).Msg("canonical url swap failed")
						} else {
							c.logger.Debug().Str("doc_id", docID).Str("from", existingURL).Str("to", item.URL).Msg("documents.url swapped to more canonical")
						}
					}
				}
				progress.Skipped.Add(1)
				embedSkipped++
				continue
			}
			if err := c.store.DeleteDocument(ctx, docID); err != nil {
				c.logger.Warn().Err(err).Str("url", item.URL).Str("doc_id", docID).Msg("delete stale document failed")
				progress.EmbedFail.Add(1)
				embedFailed++
				c.recordFailedPage(ctx, item.Source.ID, item.URL, err)
				continue
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
			// Chunking failures (e.g. tokenizer quirks on malformed sections) are
			// document-specific and shouldn't trip the consecutive-embed-fail circuit
			// breaker — the backend is healthy, the document just can't be processed.
			chunkErr := errors.Is(err, rag.ErrChunking)
			if chunkErr {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("chunking failed")
			} else {
				c.logger.Warn().Err(err).Str("url", item.URL).Msg("ingest failed")
			}
			progress.EmbedFail.Add(1)
			embedFailed++
			c.recordFailedPage(ctx, item.Source.ID, item.URL, err)
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

// prefixContentLinesWithMarker prefixes every non-empty, non-heading line of
// text with the given marker (followed by a space). Used to densely embed
// image markers into document content so every chunk produced by the
// structure-aware chunker carries at least one marker — necessary because
// chunks may slice mid-page and a once-per-page marker can land outside the
// chunked window.
//
// Headings (# / ## / ###) are passed through untouched so they remain valid
// chunk boundaries. Markers strip cleanly via markers.Strip before embedding.
func prefixContentLinesWithMarker(text, marker string) string {
	var sb strings.Builder
	sb.Grow(len(text) + len(marker)*8)
	for line := range strings.SplitSeq(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			sb.WriteByte('\n')
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			sb.WriteString(line)
			sb.WriteByte('\n')
			continue
		}
		sb.WriteString(marker)
		sb.WriteByte(' ')
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	return strings.TrimRight(sb.String(), "\n")
}

// storeArtifacts uploads each rendered image to RustFS and records a row in
// document_artifacts. Failures are logged but never propagate — text-only
// retrieval remains correct without images.
func (c *Crawler) storeArtifacts(ctx context.Context, pageURL, docID string, arts []ParsedArtifact) {
	for _, art := range arts {
		key := imageObjectStoreKey(pageURL, art.Ref)
		if err := c.objStore.Put(ctx, key, art.Data, art.MimeType); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Str("ref", art.Ref).Msg("artifact put failed")
			continue
		}
		row := &domain.Artifact{
			DocID:          docID,
			Ref:            art.Ref,
			Kind:           domain.ArtifactKind(art.Kind),
			ObjectStoreKey: key,
			MimeType:       art.MimeType,
		}
		if err := c.store.UpsertArtifact(ctx, row); err != nil {
			c.logger.Warn().Err(err).Str("url", pageURL).Str("ref", art.Ref).Msg("artifact upsert failed")
		}
	}
}
