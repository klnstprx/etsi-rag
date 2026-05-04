package store

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"etsi-rag/internal/domain"
)

// newStoreWithMutableDim builds a store whose expected embedding dimension is
// driven by a returned setter, so a single test can simulate the
// embedding-model-change flow that mutates Embedding.Dimension at runtime.
func newStoreWithMutableDim(t *testing.T, initial int) (*SQLiteStore, func(int)) {
	t.Helper()
	db, err := OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening test database: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := Migrate(db); err != nil {
		t.Fatalf("running migrations: %v", err)
	}
	var dim atomic.Int32
	dim.Store(int32(initial))
	s := NewSQLiteStore(db, func() int { return int(dim.Load()) })
	return s, func(v int) { dim.Store(int32(v)) }
}

// seedSearchCorpus creates a small corpus suitable for parity / write-through
// tests. Returns the ordered chunk content strings that match the embeddings
// inserted, so callers can assert ranking by text rather than by chunk ID.
func seedSearchCorpus(t *testing.T, s *SQLiteStore) {
	t.Helper()
	ctx := context.Background()

	docs := []*domain.Document{
		{ID: "doc-a", Title: "Alpha", Content: "alpha", Language: domain.LanguageES, Category: domain.CategoryGeneral},
		{ID: "doc-b", Title: "Bravo", Content: "bravo", Language: domain.LanguageES, Category: domain.CategoryGeneral},
		{ID: "doc-c", Title: "Charlie", Content: "charlie", Language: domain.LanguageEN, Category: domain.CategoryGeneral},
	}
	for _, d := range docs {
		if err := s.CreateDocument(ctx, d); err != nil {
			t.Fatalf("creating %s: %v", d.ID, err)
		}
	}

	type chunkSpec struct {
		docID string
		chunk domain.Chunk
		emb   []float32
	}
	specs := []chunkSpec{
		{"doc-a", domain.Chunk{Content: "a0", ChunkIndex: 0}, []float32{1.0, 0.0, 0.0, 0.0}},
		{"doc-a", domain.Chunk{Content: "a1", ChunkIndex: 1}, []float32{0.9, 0.1, 0.0, 0.0}},
		{"doc-b", domain.Chunk{Content: "b0", ChunkIndex: 0}, []float32{0.0, 1.0, 0.0, 0.0}},
		{"doc-c", domain.Chunk{Content: "c0", ChunkIndex: 0}, []float32{0.0, 0.0, 1.0, 0.0}},
		{"doc-c", domain.Chunk{Content: "c1", ChunkIndex: 1}, []float32{0.5, 0.5, 0.0, 0.0}},
	}
	for _, sp := range specs {
		if err := s.CreateChunks(ctx, sp.docID,
			[]domain.Chunk{sp.chunk},
			[][]float32{sp.emb},
		); err != nil {
			t.Fatalf("creating chunk %s/%d: %v", sp.docID, sp.chunk.ChunkIndex, err)
		}
	}
}

// resultIDs returns each result's ChunkID in order. Comparing IDs (not scores)
// is enough for parity — float ordering can be perturbed by minor numerical
// differences, but the legacy and cache paths use the exact same
// cosineSimilarity function, so chunk-ID order must match.
func resultIDs(rs []SearchResult) []int64 {
	ids := make([]int64, len(rs))
	for i, r := range rs {
		ids[i] = r.Chunk.ID
	}
	return ids
}

// TestCreateChunks_AppendsToCache verifies that CreateChunks calls made
// after Warm are reflected in subsequent SearchSimilar results without a
// re-Warm. Without write-through, post-Warm inserts would be invisible to
// the cache-backed search path.
func TestCreateChunks_AppendsToCache(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	// Warm an empty store so the cache is loaded with zero entries.
	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming empty store: %v", err)
	}

	doc := &domain.Document{
		ID: "doc-x", Title: "X", Content: "x",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-x",
		[]domain.Chunk{{Content: "x0", ChunkIndex: 0}},
		[][]float32{{1.0, 0.0, 0.0, 0.0}},
	); err != nil {
		t.Fatal(err)
	}

	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].Chunk.DocID != "doc-x" {
		t.Errorf("expected doc-x, got %s", results[0].Chunk.DocID)
	}
}

// TestDeleteDocument_InvalidatesCache verifies that a deleted document's
// chunks no longer appear in cache-backed SearchSimilar results. The DB
// CASCADE handles the row deletion; the cache hook in DeleteDocument must
// drop the matching entries.
func TestDeleteDocument_InvalidatesCache(t *testing.T) {
	s := setupTestStore(t)
	seedSearchCorpus(t, s)

	ctx := context.Background()
	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	if err := s.DeleteDocument(ctx, "doc-a"); err != nil {
		t.Fatalf("deleting doc-a: %v", err)
	}

	// Query that would have been dominated by doc-a's vectors.
	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, r := range results {
		if r.Chunk.DocID == "doc-a" {
			t.Errorf("doc-a chunk %d still surfaced after delete", r.Chunk.ID)
		}
	}
}

// TestPurgeOrphanDocuments_InvalidatesCache verifies that PurgeOrphanDocuments
// drops the cached vectors of documents it deletes. Without the hook, the
// cache would keep returning chunks whose backing rows no longer exist.
func TestPurgeOrphanDocuments_InvalidatesCache(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	src := &domain.CrawlSource{Name: "test", SeedURL: "https://example.com/", Category: "general", Language: "es", Enabled: true}
	if err := s.CreateCrawlSource(ctx, src); err != nil {
		t.Fatal(err)
	}

	// doc-orphan's embedding is closer to the query than doc-keep's, so
	// without cache invalidation the cache would still rank it first and the
	// metadata IN-fetch would silently drop it, leaving 0 results.
	specs := []struct {
		id  string
		emb []float32
	}{
		{"doc-keep", []float32{0.5, 0.5, 0.0, 0.0}},
		{"doc-orphan", []float32{1.0, 0.0, 0.0, 0.0}},
	}
	for _, sp := range specs {
		if err := s.CreateDocument(ctx, &domain.Document{
			ID: sp.id, Title: sp.id, Content: "c",
			Language: domain.LanguageES, Category: domain.CategoryGeneral,
		}); err != nil {
			t.Fatal(err)
		}
		if err := s.CreateChunks(ctx, sp.id,
			[]domain.Chunk{{Content: sp.id + "-chunk", ChunkIndex: 0}},
			[][]float32{sp.emb},
		); err != nil {
			t.Fatal(err)
		}
	}

	keepID := "doc-keep"
	now := time.Now()
	if err := s.UpsertCrawlPage(ctx, &domain.CrawlPage{
		SourceID: src.ID, URL: "https://example.com/keep", DocID: &keepID,
		ContentHash: "h", Status: domain.CrawlPageFetched, LastCrawledAt: &now,
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	purged, err := s.PurgeOrphanDocuments(ctx)
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if purged != 1 {
		t.Fatalf("expected 1 doc purged, got %d", purged)
	}

	// topK=1 deliberately small: if the cache still ranks doc-orphan at the
	// top, the metadata IN-fetch drops it silently and we'd see 0 results
	// instead of doc-keep. The cache invalidation hook is what makes
	// doc-keep visible at top-1.
	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1 (stale cache entries leaked)", len(results))
	}
	if results[0].Chunk.DocID != "doc-keep" {
		t.Errorf("expected doc-keep, got %s", results[0].Chunk.DocID)
	}
}

// TestPurgeStaleDocuments_InvalidatesCache mirrors the orphan test for the
// stale-purge path.
func TestPurgeStaleDocuments_InvalidatesCache(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	src := &domain.CrawlSource{Name: "test", SeedURL: "https://example.com/", Category: "general", Language: "es", Enabled: true}
	if err := s.CreateCrawlSource(ctx, src); err != nil {
		t.Fatal(err)
	}

	specs := []struct {
		id  string
		emb []float32
	}{
		{"doc-fresh", []float32{0.5, 0.5, 0.0, 0.0}},
		{"doc-stale", []float32{1.0, 0.0, 0.0, 0.0}},
	}
	for _, sp := range specs {
		if err := s.CreateDocument(ctx, &domain.Document{
			ID: sp.id, Title: sp.id, Content: "c",
			Language: domain.LanguageES, Category: domain.CategoryGeneral,
		}); err != nil {
			t.Fatal(err)
		}
		if err := s.CreateChunks(ctx, sp.id,
			[]domain.Chunk{{Content: sp.id + "-chunk", ChunkIndex: 0}},
			[][]float32{sp.emb},
		); err != nil {
			t.Fatal(err)
		}
	}

	fresh, stale := "doc-fresh", "doc-stale"
	now := time.Now()
	pages := []*domain.CrawlPage{
		{SourceID: src.ID, URL: "https://example.com/fresh", DocID: &fresh, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceID: src.ID, URL: "https://example.com/stale", DocID: &stale, ContentHash: "h2", Status: domain.CrawlPageStale, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := s.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	purged, err := s.PurgeStaleDocuments(ctx)
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if purged != 1 {
		t.Fatalf("expected 1 doc purged, got %d", purged)
	}

	// See orphan test for why topK=1 catches the leak.
	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1 (stale cache entries leaked)", len(results))
	}
	if results[0].Chunk.DocID != "doc-fresh" {
		t.Errorf("expected doc-fresh, got %s", results[0].Chunk.DocID)
	}
}

// TestConcurrent_SearchAndCreateChunks runs many search readers alongside a
// few CreateChunks writers to flush out data races against the shared cache
// slice. Run with `go test -race`. The body asserts only that no operation
// errors and no race is reported — result correctness is covered by the
// other tests.
//
// Uses a file-backed DB instead of :memory: because modernc.org/sqlite gives
// each connection its own private database for `:memory:`, which would make
// the writer's inserts invisible to the readers (a fixture problem unrelated
// to concurrency).
//
// SetMaxOpenConns(1) serializes DB access so the test focuses on the cache
// mutex (the actual unit under test), not on SQLite's own write-lock
// behaviour. Production callers serialise CreateChunks per document anyway
// (see rag.Pipeline.Ingest); this fixture matches that reality.
func TestConcurrent_SearchAndCreateChunks(t *testing.T) {
	dbPath := t.TempDir() + "/race.db"
	db, err := OpenDB(dbPath)
	if err != nil {
		t.Fatalf("opening DB: %v", err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if err := Migrate(db); err != nil {
		t.Fatalf("migrating: %v", err)
	}
	s := NewSQLiteStore(db, func() int { return testEmbeddingDimension })

	ctx := context.Background()

	// Seed enough rows that the search loop has real work to do.
	doc := &domain.Document{
		ID: "doc-seed", Title: "seed", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	const seedChunks = 64
	chunks := make([]domain.Chunk, seedChunks)
	embs := make([][]float32, seedChunks)
	for i := range chunks {
		chunks[i] = domain.Chunk{Content: fmt.Sprintf("c%d", i), ChunkIndex: i}
		embs[i] = []float32{float32(i % 4), 1, 0, 0}
	}
	if err := s.CreateChunks(ctx, "doc-seed", chunks, embs); err != nil {
		t.Fatal(err)
	}

	// One document per writer so each writer's chunks have a unique doc_id +
	// chunk_index and don't collide on the UNIQUE constraint.
	const writers = 2
	writerDocs := make([]string, writers)
	for i := range writerDocs {
		id := fmt.Sprintf("doc-w%d", i)
		writerDocs[i] = id
		if err := s.CreateDocument(ctx, &domain.Document{
			ID: id, Title: id, Content: "c",
			Language: domain.LanguageES, Category: domain.CategoryGeneral,
		}); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	const (
		readers      = 8
		opsPerReader = 200
		opsPerWriter = 100
	)
	query := []float32{1.0, 0.0, 0.0, 0.0}

	var wg sync.WaitGroup
	errCh := make(chan error, readers+writers)
	wg.Add(readers + writers)

	for r := range readers {
		go func(r int) {
			defer wg.Done()
			for range opsPerReader {
				if _, err := s.SearchSimilar(ctx, query, 10); err != nil {
					errCh <- fmt.Errorf("reader %d: %w", r, err)
					return
				}
			}
		}(r)
	}

	for w := range writers {
		go func(w int) {
			defer wg.Done()
			var idx atomic.Int32
			docID := writerDocs[w]
			for range opsPerWriter {
				i := int(idx.Add(1))
				err := s.CreateChunks(ctx, docID,
					[]domain.Chunk{{Content: fmt.Sprintf("w%d-%d", w, i), ChunkIndex: i}},
					[][]float32{{float32(i), 0, 1, 0}},
				)
				if err != nil {
					errCh <- fmt.Errorf("writer %d: %w", w, err)
					return
				}
			}
		}(w)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}

// TestCreateChunks_DimensionMismatchDoesNotPoisonCache verifies that a failed
// CreateChunks (rejected for wrong embedding dimension) leaves the cache
// untouched. The existing dim check at chunks.go:79 fires before the DB
// insert, so the cache append is never reached — but if that ordering ever
// changes, this test catches the regression by asserting that a subsequent
// SearchSimilar still produces correct results.
func TestCreateChunks_DimensionMismatchDoesNotPoisonCache(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-1", Title: "T", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}

	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	// Wrong dim — must error and leave the cache empty.
	err := s.CreateChunks(ctx, "doc-1",
		[]domain.Chunk{{Content: "bad", ChunkIndex: 0}},
		[][]float32{{1.0, 0.0}}, // 2 dims vs. expected 4
	)
	if err == nil {
		t.Fatal("expected dim mismatch error")
	}

	// Searching after a rejected insert should yield no results — if the
	// cache was poisoned, we'd get a hit with a chunk_id the DB has never
	// heard of, and the metadata fetch would silently swallow it.
	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 5)
	if err != nil {
		t.Fatalf("search after rejected insert: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}

	// And a correctly-dimensioned insert afterward must work end-to-end.
	if err := s.CreateChunks(ctx, "doc-1",
		[]domain.Chunk{{Content: "good", ChunkIndex: 0}},
		[][]float32{{1.0, 0.0, 0.0, 0.0}},
	); err != nil {
		t.Fatalf("good insert: %v", err)
	}
	results, err = s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 5)
	if err != nil {
		t.Fatalf("search after good insert: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results after good insert, want 1", len(results))
	}
}

// TestCreateChunks_DimChangeReprocess simulates the
// embedding-model-change-then-reprocess flow that produced
// "appending to cache: cache dim mismatch". The cache is warmed at dim=4,
// then the configured dim is bumped to dim=8 (as if the operator switched
// embedding models), and CreateChunks runs at the new dim. Before the fix,
// appendChunks compared the new vector against the stale cache.dim and
// returned an error — failing every page in reprocess. After the fix, the
// cache invalidates itself and CreateChunks succeeds; the SQL fallback
// covers SearchSimilar until a re-Warm rebuilds the cache.
func TestCreateChunks_DimChangeReprocess(t *testing.T) {
	s, setDim := newStoreWithMutableDim(t, 4)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-old", Title: "Old", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-old",
		[]domain.Chunk{{Content: "old", ChunkIndex: 0}},
		[][]float32{{1, 0, 0, 0}},
	); err != nil {
		t.Fatalf("seed insert: %v", err)
	}
	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming: %v", err)
	}

	// Operator changes embedding model — new dim is 8.
	setDim(8)

	// Reprocess: delete the stale doc, then re-ingest at the new dim.
	if err := s.DeleteDocument(ctx, "doc-old"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	doc2 := &domain.Document{
		ID: "doc-new", Title: "New", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc2); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-new",
		[]domain.Chunk{{Content: "new", ChunkIndex: 0}},
		[][]float32{{1, 0, 0, 0, 0, 0, 0, 0}},
	); err != nil {
		t.Fatalf("post-model-change insert: %v", err)
	}

	// Cache should have invalidated itself (loaded=false) so SearchSimilar
	// falls back to the SQL path, which reads authoritative DB state at the
	// new dim.
	results, err := s.SearchSimilar(ctx, []float32{1, 0, 0, 0, 0, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("search after model change: %v", err)
	}
	if len(results) != 1 || results[0].Chunk.DocID != "doc-new" {
		t.Fatalf("got %d results, want 1 from doc-new", len(results))
	}

	// Re-warm should restore the cache at the new dim and keep working.
	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("re-warm: %v", err)
	}
	results, err = s.SearchSimilar(ctx, []float32{1, 0, 0, 0, 0, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("search after re-warm: %v", err)
	}
	if len(results) != 1 || results[0].Chunk.DocID != "doc-new" {
		t.Fatalf("post-warm: got %d results, want 1 from doc-new", len(results))
	}
}

// TestSearchSimilar_ParityWithLegacy locks the contract that the cache-backed
// search returns the same ranked top-K as the legacy SQL path. The test runs
// SearchSimilar before Warm (legacy path) and after Warm (cache path) against
// an identical corpus and asserts identical chunk-ID ordering.
func TestSearchSimilar_ParityWithLegacy(t *testing.T) {
	s := setupTestStore(t)
	seedSearchCorpus(t, s)

	ctx := context.Background()
	query := []float32{0.8, 0.2, 0.0, 0.0}

	legacy, err := s.SearchSimilar(ctx, query, 5)
	if err != nil {
		t.Fatalf("legacy search: %v", err)
	}
	if len(legacy) != 5 {
		t.Fatalf("legacy: got %d results, want 5", len(legacy))
	}

	if _, err := s.Warm(ctx); err != nil {
		t.Fatalf("warming cache: %v", err)
	}

	cached, err := s.SearchSimilar(ctx, query, 5)
	if err != nil {
		t.Fatalf("cached search: %v", err)
	}
	if len(cached) != 5 {
		t.Fatalf("cached: got %d results, want 5", len(cached))
	}

	legacyIDs := resultIDs(legacy)
	cachedIDs := resultIDs(cached)
	for i := range legacyIDs {
		if legacyIDs[i] != cachedIDs[i] {
			t.Errorf("rank %d: legacy ChunkID=%d, cached ChunkID=%d", i, legacyIDs[i], cachedIDs[i])
		}
	}
}
