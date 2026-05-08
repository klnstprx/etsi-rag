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

// GetDocumentByURL is the entry point for the read_document agent tool: the
// LLM passes back a URL it saw in a previous citation, and we need to resolve
// it to a doc record. Lookup is exact-match on documents.url.
func TestGetDocumentByURL(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	url := "https://example.com/curso/info"
	doc := &domain.Document{
		ID:       "doc-with-url",
		Title:    "Curso de prueba",
		Content:  "Contenido completo del documento",
		URL:      &url,
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	t.Run("hit", func(t *testing.T) {
		got, err := s.GetDocumentByURL(ctx, url)
		if err != nil {
			t.Fatalf("GetDocumentByURL: %v", err)
		}
		if got == nil {
			t.Fatal("expected document, got nil")
		}
		if got.ID != "doc-with-url" {
			t.Errorf("ID mismatch: got %q, want doc-with-url", got.ID)
		}
		if got.Title != doc.Title {
			t.Errorf("Title mismatch: got %q, want %q", got.Title, doc.Title)
		}
		if got.Content != doc.Content {
			t.Errorf("Content mismatch: got %q, want %q", got.Content, doc.Content)
		}
	})

	t.Run("miss returns ErrNotFound", func(t *testing.T) {
		got, err := s.GetDocumentByURL(ctx, "https://example.com/does/not/exist")
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("expected ErrNotFound, got: %v", err)
		}
		if got != nil {
			t.Error("expected nil document on miss")
		}
	})

	t.Run("doc with NULL url is not returned by URL lookup", func(t *testing.T) {
		// The LLM never sees citations for URL-less docs, so by-URL lookup
		// must not surface them — and certainly not via "" matching NULL.
		nullDoc := &domain.Document{
			ID:       "doc-no-url",
			Title:    "URL-less doc",
			Content:  "x",
			URL:      nil,
			Language: domain.LanguageES,
		}
		if err := s.CreateDocument(ctx, nullDoc); err != nil {
			t.Fatalf("creating URL-less document: %v", err)
		}
		got, err := s.GetDocumentByURL(ctx, "")
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("empty URL must miss, got err=%v doc=%v", err, got)
		}
	})
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
	}
	docEN := &domain.Document{
		ID:       "doc-en",
		Title:    "English Doc",
		Content:  "Content in English",
		Language: domain.LanguageEN,
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
		Language: domain.LanguageES,
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
			Language: domain.LanguageES,
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
		URL: &url, Language: domain.LanguageEN,
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
	if r.Document.Content != "" {
		t.Errorf("doc content should be empty (not fetched), got %d bytes", len(r.Document.Content))
	}
}

func TestSearchSimilar_MultipleChunksSameDocument(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-1", Title: "Big Doc", Content: "c",
		Language: domain.LanguageES,
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
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating doc: %v", err)
	}
	docNoURL := &domain.Document{
		ID: "doc-no-url", Title: "T", Content: "x",
		Language: domain.LanguageES,
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
		Language: domain.LanguageES,
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

	const srcKey = "test"

	// Three docs: referenced, orphan, shared-by-two.
	docs := []*domain.Document{
		{ID: "doc-ref", Title: "referenced", Content: "x", Language: domain.LanguageES},
		{ID: "doc-orphan", Title: "orphan", Content: "x", Language: domain.LanguageES},
		{ID: "doc-shared", Title: "shared", Content: "x", Language: domain.LanguageES},
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
		{SourceKey: srcKey, URL: "https://example.com/a", DocID: &refID, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: srcKey, URL: "https://example.com/b", DocID: &sharedID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: srcKey, URL: "https://example.com/c", DocID: &sharedID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
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

// setupSourceKeyFilterFixture seeds two source keys ("alpha", "beta"), one
// document per source plus a third document referenced by both, and one chunk
// per document with deterministic embeddings along distinct axes. Returns the
// store ready for SearchSimilar / SearchKeyword scoping assertions.
func setupSourceKeyFilterFixture(t *testing.T) *SQLiteStore {
	t.Helper()
	s := setupTestStore(t)
	ctx := context.Background()

	docs := []*domain.Document{
		{ID: "doc-alpha", Title: "Alpha", Content: "alpha unique-alpha-token", Language: domain.LanguageES},
		{ID: "doc-beta", Title: "Beta", Content: "beta unique-beta-token", Language: domain.LanguageES},
		{ID: "doc-shared", Title: "Shared", Content: "shared unique-shared-token", Language: domain.LanguageES},
	}
	for _, d := range docs {
		if err := s.CreateDocument(ctx, d); err != nil {
			t.Fatalf("creating %s: %v", d.ID, err)
		}
	}

	now := time.Now()
	alphaID, betaID, sharedID := "doc-alpha", "doc-beta", "doc-shared"
	pages := []*domain.CrawlPage{
		{SourceKey: "alpha", URL: "https://x/alpha", DocID: &alphaID, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "beta", URL: "https://x/beta", DocID: &betaID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "alpha", URL: "https://x/shared-from-alpha", DocID: &sharedID, ContentHash: "h3", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "beta", URL: "https://x/shared-from-beta", DocID: &sharedID, ContentHash: "h3", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := s.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upserting page %s: %v", p.URL, err)
		}
	}

	chunks := map[string]struct {
		c domain.Chunk
		v []float32
	}{
		"doc-alpha":  {domain.Chunk{Content: "alpha unique-alpha-token", ChunkIndex: 0}, []float32{1, 0, 0, 0}},
		"doc-beta":   {domain.Chunk{Content: "beta unique-beta-token", ChunkIndex: 0}, []float32{0, 1, 0, 0}},
		"doc-shared": {domain.Chunk{Content: "shared unique-shared-token", ChunkIndex: 0}, []float32{0, 0, 1, 0}},
	}
	for docID, ch := range chunks {
		if err := s.CreateChunks(ctx, docID, []domain.Chunk{ch.c}, [][]float32{ch.v}); err != nil {
			t.Fatalf("creating chunk for %s: %v", docID, err)
		}
	}
	return s
}

func TestSearchSimilar_SourceKeysFilter(t *testing.T) {
	s := setupSourceKeyFilterFixture(t)
	ctx := context.Background()
	query := []float32{0.9, 0.5, 0.5, 0} // similar to all three, descending order alpha/beta/shared

	t.Run("no filter returns all chunks", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 10)
		if err != nil {
			t.Fatalf("SearchSimilar: %v", err)
		}
		if len(results) != 3 {
			t.Errorf("got %d results, want 3", len(results))
		}
	})

	t.Run("filter by alpha only", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 10, "alpha")
		if err != nil {
			t.Fatalf("SearchSimilar: %v", err)
		}
		if len(results) != 2 {
			t.Fatalf("got %d results, want 2 (alpha + shared via alpha)", len(results))
		}
		seen := map[string]bool{}
		for _, r := range results {
			seen[r.Document.ID] = true
		}
		if !seen["doc-alpha"] || !seen["doc-shared"] {
			t.Errorf("results missing expected docs: %v", seen)
		}
		if seen["doc-beta"] {
			t.Errorf("filter leaked beta: %v", seen)
		}
	})

	t.Run("shared doc returned by either source key", func(t *testing.T) {
		resAlpha, _ := s.SearchSimilar(ctx, query, 10, "alpha")
		resBeta, _ := s.SearchSimilar(ctx, query, 10, "beta")
		alphaHasShared := false
		betaHasShared := false
		for _, r := range resAlpha {
			if r.Document.ID == "doc-shared" {
				alphaHasShared = true
			}
		}
		for _, r := range resBeta {
			if r.Document.ID == "doc-shared" {
				betaHasShared = true
			}
		}
		if !alphaHasShared || !betaHasShared {
			t.Errorf("doc-shared should appear via both keys: alpha=%v beta=%v", alphaHasShared, betaHasShared)
		}
	})

	t.Run("unknown source key returns empty without error", func(t *testing.T) {
		results, err := s.SearchSimilar(ctx, query, 10, "ghost")
		if err != nil {
			t.Fatalf("SearchSimilar: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("expected empty, got %d", len(results))
		}
	})
}

func TestSearchKeyword_SourceKeysFilter(t *testing.T) {
	s := setupSourceKeyFilterFixture(t)
	ctx := context.Background()

	t.Run("no filter matches all unique tokens", func(t *testing.T) {
		results, err := s.SearchKeyword(ctx, "unique-alpha-token unique-beta-token unique-shared-token", 10)
		if err != nil {
			t.Fatalf("SearchKeyword: %v", err)
		}
		if len(results) != 3 {
			t.Errorf("got %d results, want 3", len(results))
		}
	})

	t.Run("filter by alpha hides beta-only chunks", func(t *testing.T) {
		results, err := s.SearchKeyword(ctx, "unique-alpha-token unique-beta-token unique-shared-token", 10, "alpha")
		if err != nil {
			t.Fatalf("SearchKeyword: %v", err)
		}
		seen := map[string]bool{}
		for _, r := range results {
			seen[r.Document.ID] = true
		}
		if !seen["doc-alpha"] {
			t.Error("expected doc-alpha")
		}
		if !seen["doc-shared"] {
			t.Error("expected doc-shared via alpha")
		}
		if seen["doc-beta"] {
			t.Error("filter leaked beta")
		}
	})

	t.Run("unknown key returns empty without error", func(t *testing.T) {
		results, err := s.SearchKeyword(ctx, "unique-alpha-token", 10, "ghost")
		if err != nil {
			t.Fatalf("SearchKeyword: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("expected empty, got %d", len(results))
		}
	})
}

func TestMarkPagesStaleBySourceKey(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	now := time.Now()
	pages := []*domain.CrawlPage{
		{SourceKey: "alpha", URL: "https://example.com/a1", ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "alpha", URL: "https://example.com/a2", ContentHash: "h2", Status: domain.CrawlPageFailed, LastCrawledAt: &now},
		{SourceKey: "alpha", URL: "https://example.com/a3", ContentHash: "h3", Status: domain.CrawlPageStale, LastCrawledAt: &now},
		{SourceKey: "beta", URL: "https://example.com/b1", ContentHash: "h4", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := store.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upserting page %s: %v", p.URL, err)
		}
	}

	// Mark all alpha pages stale; the already-stale a3 must not be re-counted,
	// and beta must be untouched.
	n, err := store.MarkPagesStaleBySourceKey(ctx, "alpha")
	if err != nil {
		t.Fatalf("MarkPagesStaleBySourceKey: %v", err)
	}
	if n != 2 {
		t.Errorf("affected rows = %d, want 2 (a1 + a2; a3 already stale)", n)
	}

	stale := string(domain.CrawlPageStale)
	alphaKey := "alpha"
	got, err := store.ListCrawlPages(ctx, &alphaKey, &stale)
	if err != nil {
		t.Fatalf("ListCrawlPages: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("alpha stale pages = %d, want 3 (all alpha now stale)", len(got))
	}

	betaKey := "beta"
	betaPages, err := store.ListCrawlPages(ctx, &betaKey, nil)
	if err != nil {
		t.Fatalf("ListCrawlPages beta: %v", err)
	}
	if len(betaPages) != 1 || betaPages[0].Status != domain.CrawlPageFetched {
		t.Errorf("beta page must remain fetched, got %+v", betaPages)
	}

	// Idempotent: calling again on a key whose pages are all stale returns 0.
	n, err = store.MarkPagesStaleBySourceKey(ctx, "alpha")
	if err != nil {
		t.Fatalf("MarkPagesStaleBySourceKey second call: %v", err)
	}
	if n != 0 {
		t.Errorf("second call affected rows = %d, want 0", n)
	}

	// Unknown key is a no-op, not an error.
	n, err = store.MarkPagesStaleBySourceKey(ctx, "missing")
	if err != nil {
		t.Fatalf("MarkPagesStaleBySourceKey unknown: %v", err)
	}
	if n != 0 {
		t.Errorf("unknown key affected rows = %d, want 0", n)
	}
}

func TestMarkOrphanPagesStale(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	now := time.Now()
	pages := []*domain.CrawlPage{
		{SourceKey: "alpha", URL: "https://example.com/a1", ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "beta", URL: "https://example.com/b1", ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "gamma", URL: "https://example.com/g1", ContentHash: "h3", Status: domain.CrawlPageFailed, LastCrawledAt: &now},
		{SourceKey: "gamma", URL: "https://example.com/g2", ContentHash: "h4", Status: domain.CrawlPageStale, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := store.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upserting page %s: %v", p.URL, err)
		}
	}

	// alpha + beta are known; gamma is orphaned. g2 is already stale and must
	// not be re-counted; g1 (failed) should flip to stale.
	n, err := store.MarkOrphanPagesStale(ctx, []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("MarkOrphanPagesStale: %v", err)
	}
	if n != 1 {
		t.Errorf("affected rows = %d, want 1 (only g1)", n)
	}

	for _, key := range []string{"alpha", "beta"} {
		k := key
		fetched := string(domain.CrawlPageFetched)
		got, err := store.ListCrawlPages(ctx, &k, &fetched)
		if err != nil {
			t.Fatalf("ListCrawlPages %s: %v", k, err)
		}
		if len(got) != 1 {
			t.Errorf("%s should still have 1 fetched page, got %d", k, len(got))
		}
	}

	gammaKey := "gamma"
	stale := string(domain.CrawlPageStale)
	gammaStale, err := store.ListCrawlPages(ctx, &gammaKey, &stale)
	if err != nil {
		t.Fatalf("ListCrawlPages gamma stale: %v", err)
	}
	if len(gammaStale) != 2 {
		t.Errorf("gamma stale pages = %d, want 2", len(gammaStale))
	}

	// Empty known-keys slice: every non-stale page becomes stale.
	store2 := setupTestStore(t)
	if err := store2.UpsertCrawlPage(ctx, &domain.CrawlPage{
		SourceKey: "x", URL: "https://example.com/x", ContentHash: "hx",
		Status: domain.CrawlPageFetched, LastCrawledAt: &now,
	}); err != nil {
		t.Fatalf("upserting x: %v", err)
	}
	n, err = store2.MarkOrphanPagesStale(ctx, nil)
	if err != nil {
		t.Fatalf("MarkOrphanPagesStale empty: %v", err)
	}
	if n != 1 {
		t.Errorf("empty knownKeys affected = %d, want 1", n)
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
