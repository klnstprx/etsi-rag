package ingest

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// failingChunker returns an error on every Chunk call — exercises the
// chunking-error path.
type failingChunker struct{}

func (failingChunker) Chunk(_ context.Context, _ string) ([]chunking.Chunk, error) {
	return nil, fmt.Errorf("simulated chunking failure")
}

// mockEmbedder is a configurable Embedder for tests.
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

func setupTestIngester(t *testing.T) (*Ingester, *store.SQLiteStore, *llm.EmbedderHandle) {
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
	return NewIngester(s, h, testutil.FakeChunker{}), s, h
}

func TestIngest(t *testing.T) {
	ing, _, _ := setupTestIngester(t)
	ctx := context.Background()

	docs := []domain.Document{
		{
			ID:       "doc-1",
			Title:    "Test Doc",
			Content:  "This is test content for ingestion.",
			Language: domain.LanguageES,
		},
	}

	ingested, chunks, err := ing.Ingest(ctx, docs)
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
	ing, s, h := setupTestIngester(t)
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
	h.Set(mock)

	docs := []domain.Document{{
		ID:       "doc-img",
		Title:    "imgdoc",
		Content:  "[[image:p1]] some text body before the embed call",
		Language: domain.LanguageES,
	}}

	if _, _, err := ing.Ingest(ctx, docs); err != nil {
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
	ing, _, _ := setupTestIngester(t)
	ing.SetChunker(failingChunker{})

	_, _, err := ing.Ingest(context.Background(), []domain.Document{{
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

// TestHealthCheck_NilEmbedder verifies the unconfigured case returns the
// expected error without panicking.
func TestHealthCheck_NilEmbedder(t *testing.T) {
	ing, _, h := setupTestIngester(t)
	h.Set(nil)
	if err := ing.HealthCheck(context.Background()); err == nil {
		t.Fatal("expected error when embedder is nil")
	}
}

// TestHealthCheck_BackendReachable verifies the happy path: a working
// embedder makes HealthCheck succeed.
func TestHealthCheck_BackendReachable(t *testing.T) {
	ing, _, _ := setupTestIngester(t)
	if err := ing.HealthCheck(context.Background()); err != nil {
		t.Fatalf("HealthCheck failed with healthy embedder: %v", err)
	}
}

// TestHealthCheck_BackendUnreachable verifies the failure-mode path: an
// embedder that returns an error surfaces wrapped in HealthCheck's error.
func TestHealthCheck_BackendUnreachable(t *testing.T) {
	ing, _, h := setupTestIngester(t)
	h.Set(&mockEmbedder{
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return nil, errors.New("connection refused")
		},
		embedBatchFn: func(ctx context.Context, texts []string) ([][]float32, error) {
			return nil, errors.New("connection refused")
		},
	})
	err := ing.HealthCheck(context.Background())
	if err == nil {
		t.Fatal("expected error when embedder.Embed fails")
	}
	if !strings.Contains(err.Error(), "unreachable") {
		t.Errorf("error should mention unreachability, got: %v", err)
	}
}
