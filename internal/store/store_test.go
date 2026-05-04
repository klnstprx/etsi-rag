package store

import (
	"context"
	"errors"
	"testing"
	"time"

	"etsi-rag/internal/domain"
)

const testEmbeddingDimension = 4 // Small dimension for testing

func setupTestStore(t *testing.T) *SQLiteStore {
	t.Helper()

	db, err := OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening test database: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	if err := Migrate(db); err != nil {
		t.Fatalf("running migrations: %v", err)
	}

	return NewSQLiteStore(db, func() int { return testEmbeddingDimension })
}

func TestDocumentCRUD(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	url := "https://example.com/doc1"
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Document",
		Content:  "This is test content",
		URL:      &url,
		Language: domain.LanguageES,
		Category: domain.CategoryDegree,
	}

	t.Run("create", func(t *testing.T) {
		if err := store.CreateDocument(ctx, doc); err != nil {
			t.Fatalf("creating document: %v", err)
		}
	})

	t.Run("read", func(t *testing.T) {
		retrieved, err := store.GetDocument(ctx, "doc-1")
		if err != nil {
			t.Fatalf("getting document: %v", err)
		}
		if retrieved == nil {
			t.Fatal("expected document, got nil")
		}
		if retrieved.Title != doc.Title {
			t.Errorf("title mismatch: got %q, want %q", retrieved.Title, doc.Title)
		}
		if retrieved.Language != domain.LanguageES {
			t.Errorf("language mismatch: got %q, want %q", retrieved.Language, domain.LanguageES)
		}
	})

	t.Run("list", func(t *testing.T) {
		docs, err := store.ListDocuments(ctx, nil)
		if err != nil {
			t.Fatalf("listing documents: %v", err)
		}
		if len(docs) != 1 {
			t.Errorf("expected 1 document, got %d", len(docs))
		}
	})

	t.Run("list_with_language_filter", func(t *testing.T) {
		langES := domain.LanguageES
		docsES, err := store.ListDocuments(ctx, &langES)
		if err != nil {
			t.Fatalf("listing Spanish documents: %v", err)
		}
		if len(docsES) != 1 {
			t.Errorf("expected 1 Spanish document, got %d", len(docsES))
		}

		langEN := domain.LanguageEN
		docsEN, err := store.ListDocuments(ctx, &langEN)
		if err != nil {
			t.Fatalf("listing English documents: %v", err)
		}
		if len(docsEN) != 0 {
			t.Errorf("expected 0 English documents, got %d", len(docsEN))
		}
	})

	t.Run("delete", func(t *testing.T) {
		if err := store.DeleteDocument(ctx, "doc-1"); err != nil {
			t.Fatalf("deleting document: %v", err)
		}

		retrieved, err := store.GetDocument(ctx, "doc-1")
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("expected ErrNotFound, got: %v", err)
		}
		if retrieved != nil {
			t.Error("expected nil after deletion")
		}
	})
}

func TestGetDocumentNotFound(t *testing.T) {
	store := setupTestStore(t)

	ctx := context.Background()
	doc, err := store.GetDocument(ctx, "nonexistent")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got: %v", err)
	}
	if doc != nil {
		t.Error("expected nil document")
	}
}

func TestChunksWithEmbeddings(t *testing.T) {
	store := setupTestStore(t)

	ctx := context.Background()

	// Create parent document
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Document",
		Content:  "Full content here",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	// Create chunks with embeddings
	chunks := []domain.Chunk{
		{Content: "First chunk", ChunkIndex: 0},
		{Content: "Second chunk", ChunkIndex: 1},
	}
	embeddings := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
	}

	if err := store.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatalf("creating chunks: %v", err)
	}

	// Retrieve chunks
	retrieved, err := store.GetChunksByDocument(ctx, "doc-1")
	if err != nil {
		t.Fatalf("getting chunks: %v", err)
	}
	if len(retrieved) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(retrieved))
	}
	if retrieved[0].Content != "First chunk" {
		t.Errorf("first chunk content mismatch: %q", retrieved[0].Content)
	}
	if retrieved[1].ChunkIndex != 1 {
		t.Errorf("second chunk index mismatch: %d", retrieved[1].ChunkIndex)
	}
}

func TestVectorSearch(t *testing.T) {
	store := setupTestStore(t)

	ctx := context.Background()

	// Create documents in different languages
	docES := &domain.Document{
		ID:       "doc-es",
		Title:    "Spanish Doc",
		Content:  "Content in Spanish",
		Language: domain.LanguageES,
		Category: domain.CategoryDegree,
	}
	docEN := &domain.Document{
		ID:       "doc-en",
		Title:    "English Doc",
		Content:  "Content in English",
		Language: domain.LanguageEN,
		Category: domain.CategoryDegree,
	}

	if err := store.CreateDocument(ctx, docES); err != nil {
		t.Fatalf("creating Spanish document: %v", err)
	}
	if err := store.CreateDocument(ctx, docEN); err != nil {
		t.Fatalf("creating English document: %v", err)
	}

	// Create chunks with distinct embeddings
	chunksES := []domain.Chunk{{Content: "Spanish chunk", ChunkIndex: 0}}
	embeddingsES := [][]float32{{1.0, 0.0, 0.0, 0.0}}

	chunksEN := []domain.Chunk{{Content: "English chunk", ChunkIndex: 0}}
	embeddingsEN := [][]float32{{0.0, 1.0, 0.0, 0.0}}

	if err := store.CreateChunks(ctx, "doc-es", chunksES, embeddingsES); err != nil {
		t.Fatalf("creating Spanish chunks: %v", err)
	}
	if err := store.CreateChunks(ctx, "doc-en", chunksEN, embeddingsEN); err != nil {
		t.Fatalf("creating English chunks: %v", err)
	}

	// Cosine similarity ranks across languages without a hard filter.
	query := []float32{0.9, 0.1, 0.0, 0.0} // Closer to Spanish
	results, err := store.SearchSimilar(ctx, query, 10)
	if err != nil {
		t.Fatalf("searching: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Spanish should be first (higher cosine similarity to query)
	if results[0].Document.Language != domain.LanguageES {
		t.Errorf("expected Spanish doc first, got %s", results[0].Document.Language)
	}
	if results[0].Score <= results[1].Score {
		t.Errorf("expected first result to have higher score: %f <= %f", results[0].Score, results[1].Score)
	}
}

func TestSearchSimilar_TopKLimit(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	// Create one document with 5 chunks, each pointing in a slightly different direction.
	doc := &domain.Document{
		ID: "doc-1", Title: "Multi-chunk", Content: "content",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	chunks := []domain.Chunk{
		{Content: "chunk-0", ChunkIndex: 0},
		{Content: "chunk-1", ChunkIndex: 1},
		{Content: "chunk-2", ChunkIndex: 2},
		{Content: "chunk-3", ChunkIndex: 3},
		{Content: "chunk-4", ChunkIndex: 4},
	}
	embeddings := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.9, 0.1, 0.0, 0.0},
		{0.5, 0.5, 0.0, 0.0},
		{0.1, 0.9, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	query := []float32{1.0, 0.0, 0.0, 0.0}

	t.Run("topK less than total", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 3)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 3 {
			t.Fatalf("got %d results, want 3", len(results))
		}
	})

	t.Run("topK greater than total", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 100)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 5 {
			t.Fatalf("got %d results, want 5 (all available)", len(results))
		}
	})

	t.Run("topK zero returns empty", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 0 {
			t.Fatalf("got %d results, want 0", len(results))
		}
	})
}

func TestSearchSimilar_ScoreOrdering(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	// Three documents with orthogonal embeddings.
	docs := []struct {
		id, title string
		embedding []float32
	}{
		{"doc-a", "Alpha", []float32{1.0, 0.0, 0.0, 0.0}},
		{"doc-b", "Bravo", []float32{0.0, 1.0, 0.0, 0.0}},
		{"doc-c", "Charlie", []float32{0.0, 0.0, 1.0, 0.0}},
	}
	for _, d := range docs {
		if err := s.CreateDocument(ctx, &domain.Document{
			ID: d.id, Title: d.title, Content: "c",
			Language: domain.LanguageES, Category: domain.CategoryGeneral,
		}); err != nil {
			t.Fatal(err)
		}
		if err := s.CreateChunks(ctx, d.id,
			[]domain.Chunk{{Content: d.title + " chunk", ChunkIndex: 0}},
			[][]float32{d.embedding},
		); err != nil {
			t.Fatal(err)
		}
	}

	// Query that is mostly Alpha, a bit Bravo, no Charlie.
	query := []float32{0.8, 0.2, 0.0, 0.0}
	results, err := s.SearchSimilar(ctx, query, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}

	// Verify descending score order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: [%d].Score=%f > [%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Alpha should be first, Bravo second.
	if results[0].Document.Title != "Alpha" {
		t.Errorf("first result = %q, want Alpha", results[0].Document.Title)
	}
	if results[1].Document.Title != "Bravo" {
		t.Errorf("second result = %q, want Bravo", results[1].Document.Title)
	}
}

func TestSearchSimilar_ChunkMetadata(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	url := "https://example.com/test"
	doc := &domain.Document{
		ID: "doc-1", Title: "Test Doc", Content: "full content",
		URL: &url, Language: domain.LanguageEN, Category: domain.CategoryCourse,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}

	chunks := []domain.Chunk{
		{Content: "intro text", ChunkIndex: 0, SectionTitle: "Introduction"},
		{Content: "detail text", ChunkIndex: 1, SectionTitle: "Details"},
	}
	embeddings := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	// Query matching the first chunk.
	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}

	r := results[0]

	// Chunk fields
	if r.Chunk.Content != "intro text" {
		t.Errorf("chunk content = %q, want %q", r.Chunk.Content, "intro text")
	}
	if r.Chunk.ChunkIndex != 0 {
		t.Errorf("chunk index = %d, want 0", r.Chunk.ChunkIndex)
	}
	if r.Chunk.SectionTitle != "Introduction" {
		t.Errorf("section title = %q, want %q", r.Chunk.SectionTitle, "Introduction")
	}
	if r.Chunk.DocID != "doc-1" {
		t.Errorf("chunk doc_id = %q, want %q", r.Chunk.DocID, "doc-1")
	}

	// Document fields (d.content deliberately excluded from query)
	if r.Document.ID != "doc-1" {
		t.Errorf("doc id = %q, want %q", r.Document.ID, "doc-1")
	}
	if r.Document.Title != "Test Doc" {
		t.Errorf("doc title = %q, want %q", r.Document.Title, "Test Doc")
	}
	if r.Document.URL == nil || *r.Document.URL != url {
		t.Errorf("doc url = %v, want %q", r.Document.URL, url)
	}
	if r.Document.Language != domain.LanguageEN {
		t.Errorf("doc language = %q, want %q", r.Document.Language, domain.LanguageEN)
	}
	if r.Document.Category != domain.CategoryCourse {
		t.Errorf("doc category = %q, want %q", r.Document.Category, domain.CategoryCourse)
	}
	if r.Document.Content != "" {
		t.Errorf("doc content should be empty (not fetched), got %d bytes", len(r.Document.Content))
	}
}

func TestSearchSimilar_MultipleChunksSameDocument(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-1", Title: "Big Doc", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}

	// Two chunks from the same doc, both highly relevant.
	chunks := []domain.Chunk{
		{Content: "first section", ChunkIndex: 0},
		{Content: "second section", ChunkIndex: 1},
	}
	embeddings := [][]float32{
		{0.9, 0.1, 0.0, 0.0},
		{0.8, 0.2, 0.0, 0.0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}

	// Both should reference the same document.
	if results[0].Document.ID != "doc-1" || results[1].Document.ID != "doc-1" {
		t.Error("expected both results from doc-1")
	}
	// First chunk (index 0) should score higher.
	if results[0].Chunk.ChunkIndex != 0 {
		t.Errorf("expected chunk 0 first, got chunk %d", results[0].Chunk.ChunkIndex)
	}
}

func TestSearchSimilar_EmptyStore(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	results, err := s.SearchSimilar(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if results != nil {
		t.Errorf("expected nil results on empty store, got %d", len(results))
	}
}

func TestEmbeddingDimensionValidation(t *testing.T) {
	store := setupTestStore(t)

	ctx := context.Background()

	// Create document
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test",
		Content:  "Test content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	// Try to create chunk with wrong dimension
	chunks := []domain.Chunk{{Content: "Chunk", ChunkIndex: 0}}
	wrongEmbeddings := [][]float32{{1.0, 0.0}} // Only 2 dimensions instead of 4

	err := store.CreateChunks(ctx, "doc-1", chunks, wrongEmbeddings)
	if err == nil {
		t.Error("expected error for wrong embedding dimension")
	}

	// Try to search with wrong dimension
	wrongQuery := []float32{1.0, 0.0}
	_, err = store.SearchSimilar(ctx, wrongQuery, 10)
	if err == nil {
		t.Error("expected error for wrong query dimension")
	}
}

func TestDeleteDocumentCascadesChunks(t *testing.T) {
	store := setupTestStore(t)

	ctx := context.Background()

	// Create document with chunks
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test",
		Content:  "Test content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	chunks := []domain.Chunk{{Content: "Chunk 1", ChunkIndex: 0}}
	embeddings := [][]float32{{1.0, 0.0, 0.0, 0.0}}
	if err := store.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatalf("creating chunks: %v", err)
	}

	// Delete document
	if err := store.DeleteDocument(ctx, "doc-1"); err != nil {
		t.Fatalf("deleting document: %v", err)
	}

	// Verify chunks are deleted
	retrieved, err := store.GetChunksByDocument(ctx, "doc-1")
	if err != nil {
		t.Fatalf("getting chunks: %v", err)
	}
	if len(retrieved) != 0 {
		t.Errorf("expected 0 chunks after document deletion, got %d", len(retrieved))
	}
}

func TestDocumentExists(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test",
		Content:  "x",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	t.Run("existing id returns true", func(t *testing.T) {
		ok, err := store.DocumentExists(ctx, "doc-1")
		if err != nil {
			t.Fatalf("DocumentExists: %v", err)
		}
		if !ok {
			t.Error("expected true for existing doc-1")
		}
	})

	t.Run("missing id returns false without error", func(t *testing.T) {
		ok, err := store.DocumentExists(ctx, "nonexistent")
		if err != nil {
			t.Fatalf("DocumentExists: %v", err)
		}
		if ok {
			t.Error("expected false for nonexistent id")
		}
	})
}

func TestGetDocumentURL(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	url := "https://www.uma.es/grado-en-ingenieria-informatica/"
	doc := &domain.Document{
		ID:       "doc-with-url",
		Title:    "T",
		Content:  "x",
		URL:      &url,
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating doc: %v", err)
	}
	docNoURL := &domain.Document{
		ID: "doc-no-url", Title: "T", Content: "x",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, docNoURL); err != nil {
		t.Fatalf("creating doc-no-url: %v", err)
	}

	t.Run("returns the stored URL", func(t *testing.T) {
		got, err := store.GetDocumentURL(ctx, "doc-with-url")
		if err != nil {
			t.Fatalf("GetDocumentURL: %v", err)
		}
		if got != url {
			t.Errorf("got %q, want %q", got, url)
		}
	})
	t.Run("returns empty string for null url, no error", func(t *testing.T) {
		got, err := store.GetDocumentURL(ctx, "doc-no-url")
		if err != nil {
			t.Fatalf("GetDocumentURL: %v", err)
		}
		if got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
	t.Run("missing id returns ErrNotFound", func(t *testing.T) {
		_, err := store.GetDocumentURL(ctx, "ghost")
		if err == nil || !errors.Is(err, ErrNotFound) {
			t.Errorf("expected ErrNotFound, got %v", err)
		}
	})
}

func TestUpdateDocumentURL(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	original := "https://www.uma.es/grado-en-ingenieria-informatica/cms/menu/informacion-grado/"
	doc := &domain.Document{
		ID: "doc-1", Title: "T", Content: "x", URL: &original,
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating doc: %v", err)
	}

	t.Run("updates the url", func(t *testing.T) {
		canonical := "https://www.uma.es/grado-en-ingenieria-informatica/"
		if err := store.UpdateDocumentURL(ctx, "doc-1", canonical); err != nil {
			t.Fatalf("UpdateDocumentURL: %v", err)
		}
		got, err := store.GetDocumentURL(ctx, "doc-1")
		if err != nil {
			t.Fatalf("GetDocumentURL: %v", err)
		}
		if got != canonical {
			t.Errorf("url = %q, want %q", got, canonical)
		}
	})
	t.Run("returns ErrNotFound when doc missing", func(t *testing.T) {
		err := store.UpdateDocumentURL(ctx, "ghost", "https://x")
		if err == nil || !errors.Is(err, ErrNotFound) {
			t.Errorf("expected ErrNotFound, got %v", err)
		}
	})
}

func TestPurgeOrphanDocuments(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	src := &domain.CrawlSource{Name: "test", SeedURL: "https://example.com/", Category: "general", Language: "es", Enabled: true}
	if err := store.CreateCrawlSource(ctx, src); err != nil {
		t.Fatalf("creating source: %v", err)
	}

	// Three docs: referenced, orphan, shared-by-two.
	docs := []*domain.Document{
		{ID: "doc-ref", Title: "referenced", Content: "x", Language: domain.LanguageES, Category: domain.CategoryGeneral},
		{ID: "doc-orphan", Title: "orphan", Content: "x", Language: domain.LanguageES, Category: domain.CategoryGeneral},
		{ID: "doc-shared", Title: "shared", Content: "x", Language: domain.LanguageES, Category: domain.CategoryGeneral},
	}
	for _, d := range docs {
		if err := store.CreateDocument(ctx, d); err != nil {
			t.Fatalf("creating document %s: %v", d.ID, err)
		}
	}

	// Chunks on the orphan so we can verify cascade deletion.
	chunks := []domain.Chunk{{Content: "chunk", ChunkIndex: 0}}
	embeddings := [][]float32{{1, 0, 0, 0}}
	if err := store.CreateChunks(ctx, "doc-orphan", chunks, embeddings); err != nil {
		t.Fatalf("creating chunks: %v", err)
	}

	// Two crawl_pages: one pointing at doc-ref, two (different URLs) pointing at doc-shared.
	refID, sharedID := "doc-ref", "doc-shared"
	now := time.Now()
	pages := []*domain.CrawlPage{
		{SourceID: src.ID, URL: "https://example.com/a", DocID: &refID, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceID: src.ID, URL: "https://example.com/b", DocID: &sharedID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceID: src.ID, URL: "https://example.com/c", DocID: &sharedID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := store.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upserting page: %v", err)
		}
	}

	count, err := store.PurgeOrphanDocuments(ctx)
	if err != nil {
		t.Fatalf("PurgeOrphanDocuments: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 orphan purged, got %d", count)
	}

	if ok, _ := store.DocumentExists(ctx, "doc-orphan"); ok {
		t.Error("doc-orphan should have been purged")
	}
	if ok, _ := store.DocumentExists(ctx, "doc-ref"); !ok {
		t.Error("doc-ref should remain")
	}
	if ok, _ := store.DocumentExists(ctx, "doc-shared"); !ok {
		t.Error("doc-shared should remain (referenced by two crawl_pages)")
	}

	// Chunks + embeddings for the orphan should be gone.
	remaining, err := store.GetChunksByDocument(ctx, "doc-orphan")
	if err != nil {
		t.Fatalf("GetChunksByDocument: %v", err)
	}
	if len(remaining) != 0 {
		t.Errorf("expected orphan chunks to cascade, got %d", len(remaining))
	}
}

func TestOpenDB_CacheSizePragma(t *testing.T) {
	db, err := OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening database: %v", err)
	}
	defer db.Close()

	var cacheSize int
	if err := db.QueryRow("PRAGMA cache_size").Scan(&cacheSize); err != nil {
		t.Fatalf("querying cache_size: %v", err)
	}

	// cache_size = -4000 means 4000 KiB (negative = KiB)
	if cacheSize != -4000 {
		t.Errorf("cache_size = %d, want -4000", cacheSize)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{"orthogonal", []float32{1, 0, 0}, []float32{0, 1, 0}, 0.0},
		{"opposite", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1.0},
		{"zero vector", []float32{0, 0, 0}, []float32{1, 0, 0}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			if diff := got - tt.want; diff > 0.001 || diff < -0.001 {
				t.Errorf("cosineSimilarity(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestSerializeDeserializeFloat32(t *testing.T) {
	input := []float32{1.0, -2.5, 0.0, 3.14}
	blob := serializeFloat32(input)
	output := deserializeFloat32(blob)

	if len(output) != len(input) {
		t.Fatalf("length mismatch: got %d, want %d", len(output), len(input))
	}
	for i := range input {
		if input[i] != output[i] {
			t.Errorf("index %d: got %f, want %f", i, output[i], input[i])
		}
	}
}
