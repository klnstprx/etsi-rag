package rag

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// mockEmbedder implements llm.Embedder for testing.
type mockEmbedder struct {
	embedFn      func(ctx context.Context, text string) ([]float32, error)
	embedBatchFn func(ctx context.Context, texts []string) ([][]float32, error)
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return m.embedFn(ctx, text)
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return m.embedBatchFn(ctx, texts)
}

func setupTestPipeline(t *testing.T) (*Pipeline, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening test db: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	if err := store.Migrate(db); err != nil {
		t.Fatalf("migrating: %v", err)
	}

	s := store.NewSQLiteStore(db, func() int { return 4 })
	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{
		MaxChunkSize: 100,
		ChunkOverlap: 0,
	})

	mock := &mockEmbedder{
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.5, 0.5, 0.0, 0.0}, nil
		},
		embedBatchFn: func(ctx context.Context, texts []string) ([][]float32, error) {
			result := make([][]float32, len(texts))
			for i := range texts {
				result[i] = []float32{0.5, 0.5, 0.0, 0.0}
			}
			return result, nil
		},
	}

	p := NewPipeline(s, mock, chunker, &testutil.MockCounter{}, config.RAGConfig{
		TopK:             5,
		MaxContextTokens: 1024,
	})

	return p, s
}

func TestIngest(t *testing.T) {
	p, _ := setupTestPipeline(t)
	ctx := context.Background()

	docs := []domain.Document{
		{
			ID:       "doc-1",
			Title:    "Test Doc",
			Content:  "This is test content for ingestion.",
			Language: domain.LanguageES,
			Category: domain.CategoryGeneral,
		},
	}

	ingested, chunks, err := p.Ingest(ctx, docs)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}
	if ingested != 1 {
		t.Errorf("expected 1 ingested, got %d", ingested)
	}
	if chunks != 1 {
		t.Errorf("expected 1 chunk, got %d", chunks)
	}
}

func TestRetrieve(t *testing.T) {
	p, s := setupTestPipeline(t)
	ctx := context.Background()

	// Manually insert a document with known embedding
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Doc",
		Content:  "Relevant content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	chunks := []domain.Chunk{{Content: "Relevant content", ChunkIndex: 0}}
	embeddings := [][]float32{{0.5, 0.5, 0.0, 0.0}}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	lang := "es"
	result, err := p.Retrieve(ctx, "query", &lang, nil)
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(result.Chunks) == 0 {
		t.Fatal("expected chunks in result")
	}
	if len(result.Documents) == 0 {
		t.Fatal("expected documents in result")
	}
}

func TestBuildContext(t *testing.T) {
	p, _ := setupTestPipeline(t)

	result := &RetrievalResult{
		Chunks: []ScoredChunk{
			{Chunk: domain.Chunk{DocID: "doc-1", Content: "chunk text", ChunkIndex: 0}, Score: 0.9},
		},
		Documents: []domain.Document{
			{ID: "doc-1", Title: "Test Document"},
		},
	}

	ragCtx, err := p.BuildContext(context.Background(), result)
	if err != nil {
		t.Fatalf("BuildContext error: %v", err)
	}
	if ragCtx == "" {
		t.Fatal("expected non-empty context")
	}
	if !strings.Contains(ragCtx, "Test Document") {
		t.Error("context should contain document title")
	}
	if !strings.Contains(ragCtx, "chunk text") {
		t.Error("context should contain chunk content")
	}
}

func TestSourcesFromRetrieval(t *testing.T) {
	result := &RetrievalResult{
		Chunks: []ScoredChunk{
			{Chunk: domain.Chunk{DocID: "doc-1", Content: "text", ChunkIndex: 0}, Score: 0.9},
		},
		Documents: []domain.Document{
			{ID: "doc-1", Title: "Test Doc"},
		},
	}

	sources := SourcesFromRetrieval(result)
	if len(sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(sources))
	}
	if sources[0].Title != "Test Doc" {
		t.Errorf("expected title 'Test Doc', got %q", sources[0].Title)
	}
}
