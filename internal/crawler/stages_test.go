package crawler

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// stagesTestEmbedder returns deterministic 4-dim vectors. Used here instead of
// importing api's mockEmbedder to keep crawler tests free of cross-package deps.
type stagesTestEmbedder struct{}

func (stagesTestEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.5, 0.5, 0, 0}, nil
}

func (stagesTestEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{0.5, 0.5, 0, 0}
	}
	return out, nil
}

// newEmbedStageHarness wires a temporary SQLite store + Ingester + Crawler
// stub sufficient to drive embedStage. objStore is left nil; tests must avoid
// items with Artifacts.
func newEmbedStageHarness(t *testing.T) (*Crawler, *store.SQLiteStore) {
	t.Helper()

	// A file, not :memory:, because each pooled connection to :memory: opens
	// its own empty database, and the pipeline's stages write concurrently.
	db, err := store.OpenDB(filepath.Join(t.TempDir(), "crawler.db"))
	if err != nil {
		t.Fatalf("OpenDB: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	if err := store.Migrate(db); err != nil {
		t.Fatalf("Migrate: %v", err)
	}

	s := store.NewSQLiteStore(db, func() int { return 4 })

	counter := &testutil.MockCounter{}
	chunker := chunking.NewStructureAwareChunker(
		config.ChunkingConfig{MaxChunkSize: 64, ChunkOverlap: 0, MinChunkSize: 1},
		counter,
		512,
	)
	h := llm.NewEmbedderHandle(stagesTestEmbedder{})
	ing := ingest.New(s, h, chunker)

	c := &Crawler{
		store:    s,
		ingester: ing,
		config: config.CrawlerConfig{
			MaxEmbedFails:       3,
			ProgressLogInterval: time.Hour,
		},
		logger: zerolog.Nop(),
	}
	return c, s
}

// TestEmbedStage_Force_DeletesAndReingestsExistingDocument is the regression
// test for the reprocess-skips-everything bug: when force=true and a document
// already exists in the store, embedStage must delete the stale doc (cascading
// chunk/embedding rows) and re-ingest fresh content rather than skip.
func TestEmbedStage_Force_DeletesAndReingestsExistingDocument(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	docID := StableDocID(contentHash)
	urlStr := "https://example.com/page"

	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "stale title",
		Content:  "stale content",
		URL:      &urlStr,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}
	staleEmb := [][]float32{{0.1, 0.2, 0.3, 0.4}}
	if err := s.CreateChunks(ctx, docID, []domain.Chunk{{Content: "stale chunk", ChunkIndex: 0}}, staleEmb); err != nil {
		t.Fatalf("seed CreateChunks: %v", err)
	}

	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         urlStr,
		Title:       "fresh title",
		Text:        "fresh content body for re-embedding",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, true)

	if got := progress.Ingested.Load(); got != 1 {
		t.Errorf("Ingested = %d, want 1 (force=true must re-ingest existing doc)", got)
	}
	if got := progress.Skipped.Load(); got != 0 {
		t.Errorf("Skipped = %d, want 0 (force=true must not skip existing doc)", got)
	}
	if got := progress.EmbedFail.Load(); got != 0 {
		t.Errorf("EmbedFail = %d, want 0", got)
	}

	doc, err := s.Document(ctx, docID)
	if err != nil {
		t.Fatalf("Document after reprocess: %v", err)
	}
	if doc.Title != "fresh title" {
		t.Errorf("doc.Title = %q, want %q (re-ingest replaced stale row)", doc.Title, "fresh title")
	}

	chunks, err := s.ChunksByDocument(ctx, docID)
	if err != nil {
		t.Fatalf("ChunksByDocument: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected fresh chunks after re-ingest, got none")
	}
	for _, ch := range chunks {
		if ch.Content == "stale chunk" {
			t.Errorf("stale chunk survived re-ingest: %q", ch.Content)
		}
	}
}

// TestEmbedStage_Force_KeepsCanonicalURL is the reprocess regression for the
// 2026-05-29 URL-flip: when force=true (reprocess) re-derives a document whose
// content also lives at a /cms/menu/... sub-page, deleting and recreating the
// doc must NOT flip documents.url or its title off the canonical bare page. The
// bare URL was already canonical; a sub-page reprocessed afterwards must not
// displace it.
func TestEmbedStage_Force_KeepsCanonicalURL(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "3333333333333333333333333333333333333333333333333333333333333333"
	docID := StableDocID(contentHash)
	bareURL := "https://www.uma.es/grado-en-ingenieria-del-software/"
	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "Grado en Ingeniería del Software",
		Content:  "x",
		URL:      &bareURL,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}

	// Reprocess streams the sub-page (the bare URL is not separately re-fed
	// because it shares this content hash and collapses to the same docID).
	subPageURL := "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/"
	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         subPageURL,
		Title:       "Información del grado",
		Text:        "fresh content body for re-embedding",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, true)

	if got := progress.Ingested.Load(); got != 1 {
		t.Errorf("Ingested = %d, want 1 (force must re-ingest)", got)
	}
	doc, err := s.Document(ctx, docID)
	if err != nil {
		t.Fatalf("Document: %v", err)
	}
	if doc.URL == nil || *doc.URL != bareURL || doc.Title != "Grado en Ingeniería del Software" {
		t.Errorf("document url, title = %v, %q; want the canonical page's %q, %q (reprocess must not flip to the sub-page)",
			doc.URL, doc.Title, bareURL, "Grado en Ingeniería del Software")
	}
}

// TestEmbedStage_Force_PromotesToCanonicalURL is the inverse: when a prior bad
// reprocess left documents.url on a sub-page, reprocessing the bare URL (same
// content hash) must promote documents.url back to the canonical bare URL.
func TestEmbedStage_Force_PromotesToCanonicalURL(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "4444444444444444444444444444444444444444444444444444444444444444"
	docID := StableDocID(contentHash)
	subPageURL := "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/"
	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "Grado",
		Content:  "x",
		URL:      &subPageURL,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}

	bareURL := "https://www.uma.es/grado-en-ingenieria-del-software/"
	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         bareURL,
		Title:       "Grado",
		Text:        "fresh content body for re-embedding",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, true)

	gotURL, err := s.DocumentURL(ctx, docID)
	if err != nil {
		t.Fatalf("DocumentURL: %v", err)
	}
	if gotURL != bareURL {
		t.Errorf("documents.url = %q, want %q (reprocess must promote to canonical bare URL)", gotURL, bareURL)
	}
}

// TestEmbedStage_DedupSwapsToCanonicalURL verifies that when a content-hash
// collision happens at the no-force dedup path, embedStage updates
// documents.url to the more canonical of the two URLs. UMA's CMS serves
// identical content at /grado-en-X/ and at /grado-en-X/cms/menu/.../, and
// without this swap documents.url is whichever URL won the timing race —
// non-deterministic across crawls and breaks eval lookups against the bare
// URL. The title moves with the URL so the two describe the same page.
func TestEmbedStage_DedupSwapsToCanonicalURL(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "1111111111111111111111111111111111111111111111111111111111111111"
	docID := StableDocID(contentHash)
	// Seed: the SUB-PAGE won the timing race on a previous crawl and stamped
	// its non-canonical URL onto documents.url.
	subPageURL := "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/"
	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "Información del grado",
		Content:  "x",
		URL:      &subPageURL,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}

	// Now the bare landing URL hits the embed worker with the same content_hash.
	bareURL := "https://www.uma.es/grado-en-ingenieria-del-software/"
	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         bareURL,
		Title:       "Grado",
		Text:        "x",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, false)

	if got := progress.Skipped.Load(); got != 1 {
		t.Errorf("Skipped = %d, want 1", got)
	}
	doc, err := s.Document(ctx, docID)
	if err != nil {
		t.Fatalf("Document: %v", err)
	}
	if doc.URL == nil || *doc.URL != bareURL {
		t.Errorf("documents.url = %v, want %q (bare URL has fewer segments — should win)", doc.URL, bareURL)
	}
	if doc.Title != "Grado" {
		t.Errorf("documents.title = %q, want the canonical page's %q", doc.Title, "Grado")
	}
}

// TestEmbedStage_DedupKeepsCanonicalURL is the inverse: a sub-page arriving
// after the bare URL already won must NOT displace it.
func TestEmbedStage_DedupKeepsCanonicalURL(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "2222222222222222222222222222222222222222222222222222222222222222"
	docID := StableDocID(contentHash)
	// Seed: bare URL already in.
	bareURL := "https://www.uma.es/grado-en-ingenieria-del-software/"
	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "Grado",
		Content:  "x",
		URL:      &bareURL,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}

	subPageURL := "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/"
	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         subPageURL,
		Title:       "Grado",
		Text:        "x",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, false)

	gotURL, err := s.DocumentURL(ctx, docID)
	if err != nil {
		t.Fatalf("DocumentURL: %v", err)
	}
	if gotURL != bareURL {
		t.Errorf("documents.url = %q, want %q (deeper URL must not displace canonical)", gotURL, bareURL)
	}
}

// TestEmbedStage_NoForce_SkipsExistingDocument keeps the cross-URL dedup
// behavior intact: a regular crawl that re-encounters content already in the
// store must skip ingestion (not delete) and just record the URL.
func TestEmbedStage_NoForce_SkipsExistingDocument(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()

	contentHash := "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
	docID := StableDocID(contentHash)
	urlStr := "https://example.com/dup"

	if err := s.CreateDocument(ctx, &domain.Document{
		ID:       docID,
		Title:    "kept title",
		Content:  "kept content",
		URL:      &urlStr,
		Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("seed CreateDocument: %v", err)
	}

	in := make(chan ParseItem, 1)
	in <- ParseItem{
		URL:         urlStr,
		Title:       "would-overwrite",
		Text:        "fresh text",
		ContentHash: contentHash,
		ContentType: "html",
		SourceKey:   "test",
	}
	close(in)

	progress := &RunProgress{}
	c.embedStage(ctx, in, progress, func() {}, false)

	if got := progress.Skipped.Load(); got != 1 {
		t.Errorf("Skipped = %d, want 1 (force=false must skip existing doc)", got)
	}
	if got := progress.Ingested.Load(); got != 0 {
		t.Errorf("Ingested = %d, want 0", got)
	}
	doc, err := s.Document(ctx, docID)
	if err != nil {
		t.Fatalf("Document: %v", err)
	}
	if doc.Title != "kept title" {
		t.Errorf("doc.Title = %q, want %q (no-force must not overwrite)", doc.Title, "kept title")
	}
}

// runParseStage feeds items through parseStage and returns what it emits.
func runParseStage(t *testing.T, cfg config.CrawlerConfig, items ...CrawlItem) []ParseItem {
	t.Helper()
	c := &Crawler{config: cfg, logger: zerolog.Nop()}
	in := make(chan CrawlItem, len(items))
	for _, item := range items {
		in <- item
	}
	close(in)
	out := make(chan ParseItem, len(items))
	c.parseStage(context.Background(), in, out, &RunProgress{})
	close(out)
	var got []ParseItem
	for item := range out {
		got = append(got, item)
	}
	return got
}

// htmlItem is a parsed HTML page, with a screenshot when shot is non-nil.
func htmlItem(url, text string, shot []byte) CrawlItem {
	return CrawlItem{URL: url, ContentType: "html", HTMLParsed: &HTMLResult{Title: "t", Text: text}, Screenshot: shot}
}

// TestParseStage_ScreenshotKeepsContentHash is the regression test for a
// failed screenshot changing the page's document: the [[image:full]] marker
// must not take part in the content hash.
func TestParseStage_ScreenshotKeepsContentHash(t *testing.T) {
	text := strings.Repeat("Normativa de permanencia. ", 10)
	got := runParseStage(t, config.CrawlerConfig{MinContentLen: 200},
		htmlItem("https://www.uma.es/a/", text, []byte("png")),
		htmlItem("https://www.uma.es/a/", text, nil),
	)
	if len(got) != 2 {
		t.Fatalf("parse stage emitted %d items, want 2", len(got))
	}
	if !strings.HasPrefix(got[0].Text, markers.Format("full")) {
		t.Errorf("screenshot page text has no full marker: %q", got[0].Text)
	}
	if got[0].ContentHash != got[1].ContentHash {
		t.Errorf("content hash with screenshot %s != without %s", got[0].ContentHash, got[1].ContentHash)
	}
}

// TestParseStage_ScreenshotDoesNotPassMinLength checks that the marker does
// not count toward min_content_length.
func TestParseStage_ScreenshotDoesNotPassMinLength(t *testing.T) {
	short := strings.Repeat("x", 190)
	got := runParseStage(t, config.CrawlerConfig{MinContentLen: 200},
		htmlItem("https://www.uma.es/a/", short, []byte("png")),
	)
	if len(got) != 0 {
		t.Errorf("parse stage emitted a %d-byte page below min_content_length=200", len(got[0].Text))
	}
}
