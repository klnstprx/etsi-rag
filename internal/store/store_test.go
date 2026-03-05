package store

import (
	"context"
	"testing"

	"etsi-rag/internal/domain"
)

const testEmbeddingDimension = 4 // Small dimension for testing

func setupTestStore(t *testing.T) *SQLiteStore {
	t.Helper()

	db, err := OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening test database: %v", err)
	}

	if err := Migrate(db, testEmbeddingDimension); err != nil {
		db.Close()
		t.Fatalf("running migrations: %v", err)
	}

	return NewSQLiteStore(db, testEmbeddingDimension)
}

func TestDocumentCRUD(t *testing.T) {
	store := setupTestStore(t)
	defer store.Close()

	ctx := context.Background()

	// Create
	url := "https://example.com/doc1"
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Document",
		Content:  "This is test content",
		URL:      &url,
		Language: domain.LanguageES,
		Category: domain.CategoryDegree,
	}

	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	// Read
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

	// List
	docs, err := store.ListDocuments(ctx, nil)
	if err != nil {
		t.Fatalf("listing documents: %v", err)
	}
	if len(docs) != 1 {
		t.Errorf("expected 1 document, got %d", len(docs))
	}

	// List with language filter
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

	// Delete
	if err := store.DeleteDocument(ctx, "doc-1"); err != nil {
		t.Fatalf("deleting document: %v", err)
	}

	retrieved, err = store.GetDocument(ctx, "doc-1")
	if err != nil {
		t.Fatalf("getting deleted document: %v", err)
	}
	if retrieved != nil {
		t.Error("expected nil after deletion")
	}
}

func TestGetDocumentNotFound(t *testing.T) {
	store := setupTestStore(t)
	defer store.Close()

	ctx := context.Background()
	doc, err := store.GetDocument(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if doc != nil {
		t.Error("expected nil for nonexistent document")
	}
}

func TestChunksWithEmbeddings(t *testing.T) {
	store := setupTestStore(t)
	defer store.Close()

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
	defer store.Close()

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

	// Search without language filter
	query := []float32{0.9, 0.1, 0.0, 0.0} // Closer to Spanish
	results, err := store.SearchSimilar(ctx, query, 10, nil)
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

	// Search with language filter
	langEN := domain.LanguageEN
	resultsEN, err := store.SearchSimilar(ctx, query, 10, &langEN)
	if err != nil {
		t.Fatalf("searching with English filter: %v", err)
	}
	if len(resultsEN) != 1 {
		t.Fatalf("expected 1 English result, got %d", len(resultsEN))
	}
	if resultsEN[0].Document.Language != domain.LanguageEN {
		t.Errorf("expected English result, got %s", resultsEN[0].Document.Language)
	}
}

func TestEmbeddingDimensionValidation(t *testing.T) {
	store := setupTestStore(t)
	defer store.Close()

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
	_, err = store.SearchSimilar(ctx, wrongQuery, 10, nil)
	if err == nil {
		t.Error("expected error for wrong query dimension")
	}
}

func TestDeleteDocumentCascadesChunks(t *testing.T) {
	store := setupTestStore(t)
	defer store.Close()

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
