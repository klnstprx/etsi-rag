package crawler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/rs/zerolog"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/ingest"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/objectstore"
	"etsi-rag/internal/sources"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
	"etsi-rag/internal/tokencount"
)

// fakeS3 is a path-style S3 bucket in memory. bucketMissing answers HEAD
// with 404, like RustFS before the bucket is created; failPut rejects the
// keys it matches; puts counts the PUTs it stored.
type fakeS3 struct {
	mu            sync.Mutex
	objects       map[string][]byte
	bucketMissing bool
	failPut       func(key string) bool
	puts          int
}

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	_, key, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/")
	switch {
	case r.Method == http.MethodHead && f.bucketMissing:
		w.WriteHeader(http.StatusNotFound)
	case r.Method == http.MethodHead:
		w.WriteHeader(http.StatusOK)
	case r.Method == http.MethodPut && f.failPut != nil && f.failPut(key):
		w.WriteHeader(http.StatusForbidden)
	case r.Method == http.MethodPut:
		f.objects[key], _ = io.ReadAll(r.Body)
		f.puts++
		w.WriteHeader(http.StatusOK)
	case r.Method == http.MethodGet && f.objects[key] != nil:
		w.Write(f.objects[key])
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (f *fakeS3) putCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.puts
}

func newFakeObjectStore(t *testing.T) (*objectstore.Client, *fakeS3) {
	t.Helper()
	fake := &fakeS3{objects: map[string][]byte{}}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	client, err := objectstore.New(config.ObjectStoreConfig{
		Endpoint: srv.URL, Bucket: "test", AccessKey: "k", SecretKey: "s", Region: "us-east-1",
	})
	if err != nil {
		t.Fatalf("objectstore.New: %v", err)
	}
	return client, fake
}

// failingBatchEmbedder answers the health-check ping but fails every chunk batch.
type failingBatchEmbedder struct{ stagesTestEmbedder }

func (failingBatchEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("embed timeout")
}

// newRunHarness builds a Crawler through New over a temporary store, a fake
// bucket and srcs, ready for RunAll.
func newRunHarness(t *testing.T, cfg config.CrawlerConfig, embedder llm.Embedder, srcs []domain.CrawlSource) (*Crawler, *store.SQLiteStore, *fakeS3) {
	t.Helper()
	// A file, not :memory:, because the run's goroutines use several pool
	// connections and each in-memory connection is its own database.
	db, err := store.OpenDB(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("OpenDB: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := store.Migrate(db); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	s := store.NewSQLiteStore(db, func() int { return 4 })

	chunker := chunking.NewStructureAwareChunker(
		config.ChunkingConfig{MaxChunkSize: 64, MinChunkSize: 1}, &testutil.MockCounter{}, 512)
	ing := ingest.New(s, llm.NewEmbedderHandle(embedder), chunker)

	reg := sources.New(filepath.Join(t.TempDir(), "sources.toml"))
	if err := reg.Replace(srcs); err != nil {
		t.Fatalf("Replace sources: %v", err)
	}

	obj, fake := newFakeObjectStore(t)
	c := New(s, obj, ing, cfg, reg, zerolog.Nop())
	t.Cleanup(c.Close)
	return c, s, fake
}

// contentPage is a page whose main content is body, with links to hrefs.
func contentPage(title, body string, hrefs ...string) string {
	var links strings.Builder
	for _, h := range hrefs {
		fmt.Fprintf(&links, `<a href="%s">%s</a> `, h, h)
	}
	return fmt.Sprintf(`<html lang="es"><head><title>%s</title></head><body><nav>%s</nav>
<main><h1>%s</h1><p>%s</p></main></body></html>`, title, links.String(), title, body)
}

func latestRun(t *testing.T, s *store.SQLiteStore) domain.CrawlRun {
	t.Helper()
	runs, err := s.ListCrawlRuns(context.Background(), 1)
	if err != nil || len(runs) != 1 {
		t.Fatalf("ListCrawlRuns = %v, %v; want one run", runs, err)
	}
	return runs[0]
}

// TestRunAll_FailedRefetchKeepsDocument is a second pass that cannot fetch a
// page the first pass ingested: the page is recorded as failed in this run,
// not stale, and keeps its document and chunks, which reprocess still covers.
func TestRunAll_FailedRefetchKeepsDocument(t *testing.T) {
	var down atomic.Bool
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if down.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		io.WriteString(w, contentPage("Grado", "Plan de estudios del grado en ingeniería informática."))
	}))
	defer site.Close()
	seed := site.URL + "/grado/"
	c, s, _ := newRunHarness(t, runConfig(), stagesTestEmbedder{}, []domain.CrawlSource{{Key: "grado", Name: "Grado", SeedURL: seed, Enabled: true}})
	ctx := context.Background()

	if _, err := c.RunAll(ctx); err != nil {
		t.Fatalf("pass 1: %v", err)
	}
	first, err := s.CrawlPageByURL(ctx, seed)
	if err != nil || first == nil || first.DocID == nil {
		t.Fatalf("pass 1 page = %+v, %v; want a page with a document", first, err)
	}

	down.Store(true)
	if _, err := c.RunAll(ctx); err != nil {
		t.Fatalf("pass 2: %v", err)
	}

	page, err := s.CrawlPageByURL(ctx, seed)
	if err != nil {
		t.Fatalf("CrawlPageByURL: %v", err)
	}
	if page.Status != domain.CrawlPageFailed {
		t.Errorf("status = %s, want failed", page.Status)
	}
	if page.DocID == nil || *page.DocID != *first.DocID || page.ContentHash != first.ContentHash {
		t.Errorf("page doc_id, hash = %v, %q; want %s, %q", page.DocID, page.ContentHash, *first.DocID, first.ContentHash)
	}
	chunks, err := s.ChunksByDocument(ctx, *first.DocID)
	if err != nil || len(chunks) == 0 {
		t.Errorf("pass-1 document has %d chunks (err %v), want it kept with its chunks", len(chunks), err)
	}

	// Reprocess replays the body archived by pass 1, so the kept document is
	// re-embedded with the current model like every other one, while the page
	// still shows that its last crawl failed.
	if _, err := c.Reprocess(ctx); err != nil {
		t.Fatalf("Reprocess: %v", err)
	}
	if run := latestRun(t, s); run.PagesChanged != 1 {
		t.Errorf("reprocess re-ingested %d pages, want 1", run.PagesChanged)
	}
	page, err = s.CrawlPageByURL(ctx, seed)
	if err != nil {
		t.Fatalf("CrawlPageByURL: %v", err)
	}
	if page.Status != domain.CrawlPageFailed || page.ErrorMessage == nil {
		t.Errorf("after reprocess: status = %s, has error = %t; want failed with the pass-2 error", page.Status, page.ErrorMessage != nil)
	}

	// Once the site answers again with the same content, the page recovers.
	down.Store(false)
	if _, err := c.RunAll(ctx); err != nil {
		t.Fatalf("pass 3: %v", err)
	}
	page, err = s.CrawlPageByURL(ctx, seed)
	if err != nil {
		t.Fatalf("CrawlPageByURL: %v", err)
	}
	if page.Status != domain.CrawlPageFetched || page.ErrorMessage != nil {
		t.Errorf("after recovery: status = %s, has error = %t; want fetched with no error", page.Status, page.ErrorMessage != nil)
	}
}

// TestRunAll_GonePageDropsDocument is a second pass where the site answers 404
// for a page the first pass ingested: the page is gone, so its document is
// purged instead of being kept for a transient failure.
func TestRunAll_GonePageDropsDocument(t *testing.T) {
	var gone atomic.Bool
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if gone.Load() {
			http.NotFound(w, r)
			return
		}
		io.WriteString(w, contentPage("Normativa", "Reglamento de trabajos fin de grado derogado."))
	}))
	defer site.Close()
	seed := site.URL + "/normativa/"
	c, s, _ := newRunHarness(t, runConfig(), stagesTestEmbedder{}, []domain.CrawlSource{{Key: "normativa", Name: "Normativa", SeedURL: seed, Enabled: true}})
	ctx := context.Background()

	if _, err := c.RunAll(ctx); err != nil {
		t.Fatalf("pass 1: %v", err)
	}
	first, err := s.CrawlPageByURL(ctx, seed)
	if err != nil || first == nil || first.DocID == nil {
		t.Fatalf("pass 1 page = %+v, %v; want a page with a document", first, err)
	}

	gone.Store(true)
	if _, err := c.RunAll(ctx); err != nil {
		t.Fatalf("pass 2: %v", err)
	}

	page, err := s.CrawlPageByURL(ctx, seed)
	if err != nil {
		t.Fatalf("CrawlPageByURL: %v", err)
	}
	if page.Status != domain.CrawlPageFailed || page.DocID != nil {
		t.Errorf("page status, doc_id = %s, %v; want failed with no document", page.Status, page.DocID)
	}
	if ok, err := s.DocumentExists(ctx, *first.DocID); err != nil || ok {
		t.Errorf("document of the removed page exists = %v (err %v), want purged", ok, err)
	}
}

// TestRunAll_SeedFetchedOnceByItsSource checks that a seed other sources link
// to (the ETSI home page sits under an always-followed prefix) is fetched once
// per pass, by the source that owns it.
func TestRunAll_SeedFetchedOnceByItsSource(t *testing.T) {
	const etsiPath = "/etsi-informatica/"
	var etsiHits atomic.Int32
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == etsiPath {
			etsiHits.Add(1)
			io.WriteString(w, contentPage("ETSI", "Escuela Técnica Superior de Ingeniería Informática."))
			return
		}
		io.WriteString(w, contentPage("Grado "+r.URL.Path, "Página del grado "+r.URL.Path, "https://www.uma.es"+etsiPath))
	}))
	defer site.Close()

	// The owner is listed first; the others are each started before it often
	// enough that the old per-source seed claim fetched the page twice.
	srcs := []domain.CrawlSource{{Key: "etsi", Name: "ETSI", SeedURL: "https://www.uma.es" + etsiPath, Enabled: true}}
	for i := range 4 {
		key := fmt.Sprintf("grado-%d", i)
		srcs = append(srcs, domain.CrawlSource{Key: key, Name: key, SeedURL: "https://www.uma.es/" + key + "/", Enabled: true})
	}
	c, s, _ := newRunHarness(t, runConfig(), stagesTestEmbedder{}, srcs)
	c.fetcher.client.Transport = siteTransport{addr: site.Listener.Addr().String()}

	if _, err := c.RunAll(context.Background()); err != nil {
		t.Fatalf("RunAll: %v", err)
	}
	if n := etsiHits.Load(); n != 1 {
		t.Errorf("ETSI seed fetched %d times, want 1", n)
	}
	page, err := s.CrawlPageByURL(context.Background(), "https://www.uma.es"+etsiPath)
	if err != nil || page == nil {
		t.Fatalf("CrawlPageByURL = %v, %v", page, err)
	}
	if page.SourceKey != "etsi" {
		t.Errorf("ETSI seed recorded under source %q, want etsi", page.SourceKey)
	}
}

// TestRunAll_MissingBucketFailsBeforeRun checks that a crawl into a bucket
// that does not exist refuses to start instead of dropping every raw body
// and image.
func TestRunAll_MissingBucketFailsBeforeRun(t *testing.T) {
	c, s, fake := newRunHarness(t, runConfig(), stagesTestEmbedder{}, nil)
	fake.bucketMissing = true

	if _, err := c.RunAll(context.Background()); err == nil {
		t.Fatal("RunAll succeeded with a missing bucket")
	}
	runs, err := s.ListCrawlRuns(context.Background(), 10)
	if err != nil {
		t.Fatalf("ListCrawlRuns: %v", err)
	}
	if len(runs) != 0 {
		t.Errorf("got %d crawl_runs rows, want none", len(runs))
	}
}

// TestRunAll_EmbedFailuresFailRun checks that a run aborted by the
// max_embed_fails breaker is reported and recorded as failed, with the reason,
// rather than as cancelled or completed.
func TestRunAll_EmbedFailuresFailRun(t *testing.T) {
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, contentPage("Página", "Contenido distinto de "+r.URL.Path))
	}))
	defer site.Close()
	var srcs []domain.CrawlSource
	for i := range 3 {
		key := fmt.Sprintf("s%d", i)
		srcs = append(srcs, domain.CrawlSource{Key: key, Name: key, SeedURL: site.URL + "/" + key + "/", Enabled: true})
	}
	c, s, _ := newRunHarness(t, runConfig(), failingBatchEmbedder{}, srcs)

	if _, err := c.RunAll(context.Background()); err == nil {
		t.Error("RunAll returned no error for a run the breaker aborted")
	}
	run := latestRun(t, s)
	if run.Status != domain.CrawlRunFailed {
		t.Errorf("run status = %s, want failed", run.Status)
	}
	if run.ErrorMessage == nil || *run.ErrorMessage == "" {
		t.Error("run has no error message")
	}
}

// TestEmbedStage_TokenizerDownTripsBreaker checks that a tokenizer outage
// counts toward max_embed_fails: the stage aborts the run after the threshold
// instead of failing every remaining page one by one.
func TestEmbedStage_TokenizerDownTripsBreaker(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	tokenizer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer tokenizer.Close()
	counter := tokencount.NewLlamaCppCounter(tokenizer.URL, "m", nil).WithMaxRetries(0)
	chunker := chunking.NewStructureAwareChunker(config.ChunkingConfig{MaxChunkSize: 64, MinChunkSize: 1}, counter, 512)
	c.ingester = ingest.New(s, llm.NewEmbedderHandle(stagesTestEmbedder{}), chunker)
	c.config.MaxEmbedFails = 2

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	urls := []string{"https://www.uma.es/a/", "https://www.uma.es/b/", "https://www.uma.es/c/"}
	in := make(chan ParseItem, len(urls))
	for i, u := range urls {
		in <- ParseItem{URL: u, Title: "T", Text: "texto " + u, ContentHash: strings.Repeat(fmt.Sprint(i+5), 64), ContentType: "html", SourceKey: "t"}
	}
	close(in)

	c.embedStage(ctx, in, &RunProgress{}, cancel, false)

	if ctx.Err() == nil {
		t.Error("run not aborted after max_embed_fails tokenizer failures")
	}
	if page, _ := s.CrawlPageByURL(context.Background(), urls[2]); page != nil {
		t.Errorf("page after the abort was processed: %+v", page)
	}
}

// TestEmbedStage_RetriesMissingArtifacts checks that an image the bucket
// rejected on the first crawl is stored on the next crawl of the unchanged
// page, and that a stored image is not uploaded again.
func TestEmbedStage_RetriesMissingArtifacts(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	obj, fake := newFakeObjectStore(t)
	c.objStore = obj
	ctx := context.Background()

	hash := strings.Repeat("3", 64)
	item := ParseItem{
		URL: "https://www.uma.es/doc.pdf", Title: "Doc", Text: "[[image:p1]]\nTexto de la página uno",
		ContentHash: hash, ContentType: "pdf", SourceKey: "t",
		Artifacts: []ParsedArtifact{{Ref: "p1", Kind: domain.ArtifactPDFPage, MimeType: "image/png", Data: []byte("png")}},
	}
	embed := func() {
		in := make(chan ParseItem, 1)
		in <- item
		close(in)
		c.embedStage(ctx, in, &RunProgress{}, func() {}, false)
	}

	fake.failPut = func(key string) bool { return strings.HasPrefix(key, "crawl/img/") }
	embed()
	if _, err := s.Artifact(ctx, StableDocID(hash), "p1"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("first crawl: Artifact err = %v, want ErrNotFound", err)
	}

	fake.failPut = nil
	embed()
	if _, err := s.Artifact(ctx, StableDocID(hash), "p1"); err != nil {
		t.Errorf("second crawl: artifact p1 still missing: %v", err)
	}

	puts := fake.putCount()
	embed()
	if n := fake.putCount() - puts; n != 0 {
		t.Errorf("third crawl uploaded %d objects, want 0 (p1 is already stored)", n)
	}
}

// TestEmbedStage_ReingestsPageAfterStalePurge checks that a page whose
// document Purge Stale removed is ingested again when a later crawl finds it
// with the same content, instead of being skipped as unchanged.
func TestEmbedStage_ReingestsPageAfterStalePurge(t *testing.T) {
	c, s := newEmbedStageHarness(t)
	ctx := context.Background()
	hash := strings.Repeat("4", 64)
	item := ParseItem{URL: "https://www.uma.es/becas/", Title: "Becas", Text: "Convocatoria de becas", ContentHash: hash, ContentType: "html", SourceKey: "becas"}
	embed := func() {
		in := make(chan ParseItem, 1)
		in <- item
		close(in)
		c.embedStage(ctx, in, &RunProgress{}, func() {}, false)
	}

	embed()
	if _, err := s.MarkPagesStaleBySourceKey(ctx, "becas"); err != nil {
		t.Fatalf("MarkPagesStaleBySourceKey: %v", err)
	}
	if _, err := s.PurgeStaleDocuments(ctx); err != nil {
		t.Fatalf("PurgeStaleDocuments: %v", err)
	}

	embed()
	page, err := s.CrawlPageByURL(ctx, item.URL)
	if err != nil || page == nil || page.DocID == nil {
		t.Fatalf("page = %+v, %v; want it linked to a document", page, err)
	}
	if ok, _ := s.DocumentExists(ctx, *page.DocID); !ok {
		t.Errorf("document %s missing after the page came back", *page.DocID)
	}
}
