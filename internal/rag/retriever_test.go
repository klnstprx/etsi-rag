package rag

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// mockEmbedder implements llm.Embedder for retriever tests.
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

func defaultMockEmbedder() *mockEmbedder {
	return &mockEmbedder{
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.5, 0.5, 0.0, 0.0}, nil
		},
		embedBatchFn: func(ctx context.Context, texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{0.5, 0.5, 0.0, 0.0}
			}
			return out, nil
		},
	}
}

func setupTestRetriever(t *testing.T) (*Retriever, *store.SQLiteStore, *llm.EmbedderHandle) {
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
	h := llm.NewEmbedderHandle(defaultMockEmbedder())
	r := NewRetriever(s, h, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             5,
		MaxContextTokens: 1024,
	})
	return r, s, h
}

func TestRetrieve(t *testing.T) {
	r, s, _ := setupTestRetriever(t)
	ctx := context.Background()

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

	result, err := r.Retrieve(ctx, "query", RetrieveOpts{})
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
	r, s, _ := setupTestRetriever(t)
	ctx := context.Background()

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

	got, err := r.RetrieveInDocument(ctx, "any query", "doc-a", 5)
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
	r, _, _ := setupTestRetriever(t)
	got, err := r.RetrieveInDocument(context.Background(), "q", "missing", 5)
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
	r, _, h := setupTestRetriever(t)
	h.Set(nil)
	_, err := r.RetrieveInDocument(context.Background(), "q", "doc-a", 5)
	if err == nil {
		t.Fatal("expected error when embedder is nil")
	}
}

func TestBuildContext(t *testing.T) {
	r, _, _ := setupTestRetriever(t)

	result := &RetrievalResult{
		Chunks: []ScoredChunk{
			{Chunk: domain.Chunk{DocID: "doc-1", Content: "chunk text", ChunkIndex: 0}, Score: 0.9},
		},
		Documents: []domain.Document{
			{ID: "doc-1", Title: "Test Document"},
		},
	}

	ragCtx, err := r.BuildContext(context.Background(), result)
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

// The agent's read_document tool takes a document URL, so the search context
// the model sees must name each block's source URL.
func TestBuildContext_ShowsDocumentURL(t *testing.T) {
	r, _, _ := setupTestRetriever(t)
	url := "https://www.uma.es/grado-en-ingenieria-del-software/"

	result := &RetrievalResult{
		Chunks: []ScoredChunk{
			{Chunk: domain.Chunk{DocID: "doc-1", Content: "chunk text", SectionTitle: "Plan de estudios"}, Score: 0.9},
			{Chunk: domain.Chunk{DocID: "doc-2", Content: "other text"}, Score: 0.8},
		},
		Documents: []domain.Document{
			{ID: "doc-1", Title: "Grado UMA", URL: &url},
			{ID: "doc-2", Title: "No URL"},
		},
	}

	ragCtx, err := r.BuildContext(context.Background(), result)
	if err != nil {
		t.Fatalf("BuildContext error: %v", err)
	}
	if !strings.Contains(ragCtx, "[1] Grado UMA > Plan de estudios:\nURL: "+url+"\nchunk text") {
		t.Errorf("first block should name its URL under the header; got:\n%s", ragCtx)
	}
	if !strings.Contains(ragCtx, "[2] No URL:\nother text") {
		t.Errorf("a block without a URL keeps the plain header; got:\n%s", ragCtx)
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
