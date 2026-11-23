package ingest

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
	"etsi-rag/internal/tokencount"
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
	return New(s, h, testutil.FakeChunker{}), s, h
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

	stored, err := s.ChunksByDocument(ctx, "doc-img")
	if err != nil {
		t.Fatalf("ChunksByDocument: %v", err)
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

// downTokenizerChunker is the production chunker on a llama.cpp tokenizer
// that refuses connections.
func downTokenizerChunker(t *testing.T) chunking.Chunker {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	srv.Close()
	counter := tokencount.NewLlamaCppCounter(srv.URL, "m", nil).WithMaxRetries(0)
	return chunking.NewStructureAwareChunker(config.ChunkingConfig{MaxChunkSize: 64, MinChunkSize: 1}, counter, 512)
}

// TestIngest_TokenizerDownIsNotChunkingError checks that a tokenizer outage
// is reported as a backend failure, not a document-specific chunking error,
// so the crawler's embed-failure breaker counts it.
func TestIngest_TokenizerDownIsNotChunkingError(t *testing.T) {
	ing, _, _ := setupTestIngester(t)
	ing.SetChunker(downTokenizerChunker(t))

	_, _, err := ing.Ingest(context.Background(), []domain.Document{{
		ID: "doc-x", Content: "some text", Language: domain.LanguageES,
	}})
	if !errors.Is(err, tokencount.ErrBackend) || errors.Is(err, ErrChunking) {
		t.Errorf("err = %v, want tokencount.ErrBackend and not ErrChunking", err)
	}
}

// TestIngest_FailureLeavesNoDocument checks that a document whose chunks
// could not be embedded or stored is not left behind without chunks.
func TestIngest_FailureLeavesNoDocument(t *testing.T) {
	tests := []struct {
		name  string
		batch func(texts []string) ([][]float32, error)
	}{
		{"embedding fails", func([]string) ([][]float32, error) {
			return nil, errors.New("embed timeout")
		}},
		{"storing chunks fails", func(texts []string) ([][]float32, error) {
			// The store expects 4 dimensions, so CreateChunks rejects these.
			out := make([][]float32, len(texts))
			for i := range out {
				out[i] = []float32{1, 0}
			}
			return out, nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ing, s, h := setupTestIngester(t)
			h.Set(&mockEmbedder{
				embedFn: func(context.Context, string) ([]float32, error) { return []float32{0.5, 0.5, 0, 0}, nil },
				embedBatchFn: func(_ context.Context, texts []string) ([][]float32, error) {
					return tt.batch(texts)
				},
			})
			ctx := context.Background()

			if _, _, err := ing.Ingest(ctx, []domain.Document{{
				ID: "doc-x", Content: "some text", Language: domain.LanguageES,
			}}); err == nil {
				t.Fatal("expected an error")
			}
			if ok, err := s.DocumentExists(ctx, "doc-x"); err != nil || ok {
				t.Errorf("DocumentExists = %v, %v; want false, nil", ok, err)
			}
		})
	}
}

// TestHealthCheck_TokenizerDown checks that a run does not start while the
// chunker's tokenizer is unreachable.
func TestHealthCheck_TokenizerDown(t *testing.T) {
	ing, _, _ := setupTestIngester(t)
	ing.SetChunker(downTokenizerChunker(t))

	if err := ing.HealthCheck(context.Background()); !errors.Is(err, tokencount.ErrBackend) {
		t.Errorf("HealthCheck = %v, want tokencount.ErrBackend", err)
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

// TestIngest_TitleInSearch checks that with the toggle on, a chunk whose own
// text never names its document is found by its document title through both
// the dense and the keyword index, while the stored content stays plain body.
func TestIngest_TitleInSearch(t *testing.T) {
	for _, on := range []bool{true, false} {
		t.Run(fmt.Sprintf("on=%v", on), func(t *testing.T) {
			ing, s, h := setupTestIngester(t)
			ctx := context.Background()
			// The vector says whether the embedded text names the degree.
			vec := func(text string) []float32 {
				if strings.Contains(strings.ToLower(text), "informática") {
					return []float32{1, 0, 0, 0}
				}
				return []float32{0, 1, 0, 0}
			}
			h.Set(&mockEmbedder{
				embedFn: func(_ context.Context, text string) ([]float32, error) { return vec(text), nil },
				embedBatchFn: func(_ context.Context, texts []string) ([][]float32, error) {
					out := make([][]float32, len(texts))
					for i, text := range texts {
						out[i] = vec(text)
					}
					return out, nil
				},
			})
			ing.SetTitleInSearch(on)

			const body = "Modalidad de enseñanza Presencial. Plazas nuevo ingreso 150 plazas."
			if _, _, err := ing.Ingest(ctx, []domain.Document{{
				ID: "doc-gii", Title: "Grado en Ingeniería Informática", Content: body, Language: domain.LanguageES,
			}}); err != nil {
				t.Fatalf("Ingest: %v", err)
			}

			keyword, err := s.SearchKeyword(ctx, "informatica", 5)
			if err != nil {
				t.Fatalf("SearchKeyword: %v", err)
			}
			if got := len(keyword) == 1; got != on {
				t.Errorf("keyword search by title word found %d chunks, want found=%v", len(keyword), on)
			}
			dense, err := s.SearchSimilar(ctx, []float32{1, 0, 0, 0}, 1)
			if err != nil || len(dense) != 1 {
				t.Fatalf("SearchSimilar = %v, %v", dense, err)
			}
			if got := dense[0].Score > 0.99; got != on {
				t.Errorf("dense score against the title vector = %.2f, want match=%v", dense[0].Score, on)
			}
			if dense[0].Chunk.Content != body {
				t.Errorf("stored content = %q, want the plain body", dense[0].Chunk.Content)
			}
		})
	}
}
