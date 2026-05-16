package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// failingChunker returns an error on every Chunk call — used to exercise the
// chunking-error path in Ingest.
type failingChunker struct{}

func (failingChunker) Chunk(_ context.Context, _ string) ([]chunking.Chunk, error) {
	return nil, fmt.Errorf("simulated chunking failure")
}

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

	p := NewPipeline(s, mock, testutil.FakeChunker{}, &testutil.MockCounter{}, config.RetrievalConfig{
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

// TestIngest_StripsImageMarkersFromEmbedText asserts that image markers in
// the parser output are stripped from chunk content (so they never pollute
// the embedding space and are never user-visible) and that the chunker
// captures the corresponding refs onto the chunk's ImageRefs field for the
// vision attachment path.
func TestIngest_StripsImageMarkersFromEmbedText(t *testing.T) {
	p, s := setupTestPipeline(t)
	ctx := context.Background()

	var seenTexts []string
	mock := &mockEmbedder{
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.5, 0.5, 0.0, 0.0}, nil
		},
		embedBatchFn: func(ctx context.Context, texts []string) ([][]float32, error) {
			seenTexts = append(seenTexts, texts...)
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{0.5, 0.5, 0.0, 0.0}
			}
			return out, nil
		},
	}
	p.SetEmbedder(mock)

	docs := []domain.Document{{
		ID:       "doc-img",
		Title:    "imgdoc",
		Content:  "[[image:p1]] some text body before the embed call",
		Language: domain.LanguageES,
	}}

	if _, _, err := p.Ingest(ctx, docs); err != nil {
		t.Fatalf("Ingest: %v", err)
	}

	if len(seenTexts) == 0 {
		t.Fatal("embedder was not called")
	}
	for i, txt := range seenTexts {
		if strings.Contains(txt, "[[image:") {
			t.Errorf("embedder text[%d] still contains marker: %q", i, txt)
		}
	}

	// Stored chunk content is marker-free; the page ref lives on ImageRefs
	// (separate column) so the agent's vision path can attach the artifact
	// without re-extracting markers from text.
	stored, err := s.GetChunksByDocument(ctx, "doc-img")
	if err != nil {
		t.Fatalf("GetChunksByDocument: %v", err)
	}
	if len(stored) == 0 {
		t.Fatal("expected stored chunks")
	}
	for i, c := range stored {
		if strings.Contains(c.Content, "[[image:") {
			t.Errorf("stored chunk[%d] should be marker-free; got: %q", i, c.Content)
		}
	}
	if got := stored[0].ImageRefs; len(got) != 1 || got[0] != "p1" {
		t.Errorf("stored[0].ImageRefs = %v, want [p1]", got)
	}
}

// TestIngest_ChunkingFailureIsIdentifiable verifies that chunking failures
// surface through Ingest wrapped with ErrChunking, so callers (e.g. the crawler
// embed stage) can distinguish them from embedding failures and scope their
// circuit breakers accordingly.
func TestIngest_ChunkingFailureIsIdentifiable(t *testing.T) {
	p, _ := setupTestPipeline(t)
	p.SetChunker(failingChunker{})

	_, _, err := p.Ingest(context.Background(), []domain.Document{{
		ID:       "doc-x",
		Content:  "anything",
		Language: domain.LanguageES,
	}})
	if err == nil {
		t.Fatal("expected chunking error, got nil")
	}
	if !errors.Is(err, ErrChunking) {
		t.Errorf("expected errors.Is(err, ErrChunking), err = %v", err)
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
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	chunks := []domain.Chunk{{Content: "Relevant content", ChunkIndex: 0}}
	embeddings := [][]float32{{0.5, 0.5, 0.0, 0.0}}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	result, err := p.Retrieve(ctx, "query", RetrieveOpts{})
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

// RetrieveInDocument is the doc-scoped variant powering read_document's
// query mode. It must (a) embed the query, (b) restrict the search to the
// requested doc's chunks, (c) wrap the result identically to Retrieve so
// the caller can reuse BuildContext.
func TestRetrieveInDocument(t *testing.T) {
	p, s := setupTestPipeline(t)
	ctx := context.Background()

	// Two docs. SearchInDocument-level test already covers ranking; here we
	// only verify the pipeline correctly threads docID and embedder.
	for _, id := range []string{"doc-a", "doc-b"} {
		if err := s.CreateDocument(ctx, &domain.Document{
			ID: id, Title: "Doc " + id, Content: "c",
			Language: domain.LanguageES,
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.CreateChunks(ctx, "doc-a",
		[]domain.Chunk{{Content: "alpha chunk", ChunkIndex: 0}},
		[][]float32{{0.5, 0.5, 0, 0}},
	); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-b",
		[]domain.Chunk{{Content: "beta chunk", ChunkIndex: 0}},
		[][]float32{{0.5, 0.5, 0, 0}},
	); err != nil {
		t.Fatal(err)
	}

	got, err := p.RetrieveInDocument(ctx, "any query", "doc-a", 5)
	if err != nil {
		t.Fatalf("RetrieveInDocument: %v", err)
	}
	if len(got.Chunks) != 1 {
		t.Fatalf("expected 1 chunk (only doc-a), got %d", len(got.Chunks))
	}
	if got.Chunks[0].Chunk.DocID != "doc-a" {
		t.Errorf("got chunk from %q, want doc-a", got.Chunks[0].Chunk.DocID)
	}
	if len(got.Documents) != 1 || got.Documents[0].ID != "doc-a" {
		t.Errorf("Documents: got %v, want [doc-a]", got.Documents)
	}
}

func TestRetrieveInDocument_UnknownDocReturnsEmpty(t *testing.T) {
	p, _ := setupTestPipeline(t)
	got, err := p.RetrieveInDocument(context.Background(), "q", "missing", 5)
	if err != nil {
		t.Fatalf("RetrieveInDocument: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil RetrievalResult on miss (empty chunks ok)")
	}
	if len(got.Chunks) != 0 {
		t.Errorf("expected 0 chunks, got %d", len(got.Chunks))
	}
}

func TestRetrieveInDocument_EmbedderUnconfiguredFails(t *testing.T) {
	p, _ := setupTestPipeline(t)
	p.SetEmbedder(nil)
	_, err := p.RetrieveInDocument(context.Background(), "q", "doc-a", 5)
	if err == nil {
		t.Fatal("expected error when embedder is nil")
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
