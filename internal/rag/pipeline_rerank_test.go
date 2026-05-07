package rag

import (
	"context"
	"errors"
	"testing"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// reverseReranker scores docs by negative index — i.e. inverts the input
// order. Useful for "did the reranker actually run?" assertions: if the top
// result switches from chunk 0 to chunk N-1, rerank ran.
type reverseReranker struct {
	calls int
}

func (r *reverseReranker) Rerank(_ context.Context, _ string, docs []string, _ int) ([]llm.RerankResult, error) {
	r.calls++
	out := make([]llm.RerankResult, len(docs))
	for i := range docs {
		out[i] = llm.RerankResult{Index: len(docs) - 1 - i, Score: float32(len(docs) - i)}
	}
	return out, nil
}

// erroringReranker always errors. Pipeline must fall through to the
// pre-rerank ordering, never bubble the error up to the caller.
type erroringReranker struct {
	calls int
}

func (e *erroringReranker) Rerank(_ context.Context, _ string, _ []string, _ int) ([]llm.RerankResult, error) {
	e.calls++
	return nil, errors.New("rerank backend down")
}

// recordingReranker captures the docs it received so callers can assert on
// over-fetch / candidate-pool sizing.
type recordingReranker struct {
	seen []string
}

func (r *recordingReranker) Rerank(_ context.Context, _ string, docs []string, _ int) ([]llm.RerankResult, error) {
	r.seen = append([]string{}, docs...)
	// Echo back the input order so candidates flow through unchanged.
	out := make([]llm.RerankResult, len(docs))
	for i := range docs {
		out[i] = llm.RerankResult{Index: i, Score: float32(len(docs) - i)}
	}
	return out, nil
}

// setupDensePipeline seeds 5 chunks in dense mode with cosine scores
// monotonically decreasing from chunk 0 (best) to chunk 4 (worst).
func setupDensePipeline(t *testing.T, topK int) (*Pipeline, *store.SQLiteStore) {
	t.Helper()
	ctx := context.Background()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := store.Migrate(db); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	s := store.NewSQLiteStore(db, func() int { return 4 })

	doc := &domain.Document{
		ID: "doc-1", Title: "Test", Content: "joined",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("create doc: %v", err)
	}

	// Query embedder returns [1, 0, 0, 0]. Each chunk's score is its first
	// component, so chunk i has score (5-i)/5 — monotonic decreasing.
	chunks := []domain.Chunk{
		{Content: "chunk zero", ChunkIndex: 0},
		{Content: "chunk one", ChunkIndex: 1},
		{Content: "chunk two", ChunkIndex: 2},
		{Content: "chunk three", ChunkIndex: 3},
		{Content: "chunk four", ChunkIndex: 4},
	}
	embeddings := [][]float32{
		{1.0, 0, 0, 0},
		{0.8, 0.6, 0, 0},
		{0.6, 0.8, 0, 0},
		{0.4, 0.9, 0, 0},
		{0.2, 0.95, 0, 0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatalf("create chunks: %v", err)
	}

	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{
		MaxChunkSize: 100,
		ChunkOverlap: 0,
	})
	emb := fixedEmbedder{vec: []float32{1.0, 0, 0, 0}}

	p := NewPipeline(s, emb, chunker, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             topK,
		MaxContextTokens: 1024,
		RetrievalMode:    "dense",
	})
	return p, s
}

func TestRetrieve_RerankInvertsOrder(t *testing.T) {
	p, _ := setupDensePipeline(t, 5)
	rr := &reverseReranker{}
	p.SetReranker(rr, 5)

	got, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if rr.calls != 1 {
		t.Errorf("reranker calls = %d, want 1", rr.calls)
	}
	if len(got.Chunks) != 5 {
		t.Fatalf("got %d chunks, want 5", len(got.Chunks))
	}
	// Pre-rerank order: 0,1,2,3,4. Reverse reranker flips it: 4,3,2,1,0.
	want := []int{4, 3, 2, 1, 0}
	for i, w := range want {
		if got.Chunks[i].Chunk.ChunkIndex != w {
			var ranks []int
			for _, c := range got.Chunks {
				ranks = append(ranks, c.Chunk.ChunkIndex)
			}
			t.Errorf("chunk order = %v, want %v", ranks, want)
			break
		}
	}
}

func TestRetrieve_RerankErrorFallsThrough(t *testing.T) {
	p, _ := setupDensePipeline(t, 3)
	er := &erroringReranker{}
	p.SetReranker(er, 5)

	got, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve must not surface reranker errors, got: %v", err)
	}
	if er.calls != 1 {
		t.Errorf("erroring reranker calls = %d, want 1", er.calls)
	}
	// Original dense order should be preserved (chunk 0 best, descending).
	want := []int{0, 1, 2}
	for i, w := range want {
		if got.Chunks[i].Chunk.ChunkIndex != w {
			t.Errorf("chunk[%d] = %d, want %d (pre-rerank order)", i, got.Chunks[i].Chunk.ChunkIndex, w)
		}
	}
}

func TestRetrieve_NilRerankerSkipsRerank(t *testing.T) {
	p, _ := setupDensePipeline(t, 3)
	// No SetReranker call — reranker is nil. Retrieve should behave exactly
	// like the pre-rerank pipeline.
	got, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(got.Chunks) != 3 {
		t.Fatalf("got %d chunks, want 3", len(got.Chunks))
	}
	if got.Chunks[0].Chunk.ChunkIndex != 0 {
		t.Errorf("nil reranker should not reorder; top chunk = %d, want 0",
			got.Chunks[0].Chunk.ChunkIndex)
	}
}

func TestRetrieve_RerankOverFetchesCandidatePool(t *testing.T) {
	// topK=2, but reranker pool=5 — Retrieve must hand the reranker 5 docs,
	// not 2, so the cross-encoder has room to swap.
	p, _ := setupDensePipeline(t, 2)
	rec := &recordingReranker{}
	p.SetReranker(rec, 5)

	got, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(rec.seen) != 5 {
		t.Errorf("reranker saw %d docs, want 5 (over-fetch from pool)", len(rec.seen))
	}
	// After rerank (passthrough order), trim to topK=2.
	if len(got.Chunks) != 2 {
		t.Errorf("trimmed result len = %d, want 2", len(got.Chunks))
	}
}

func TestRetrieve_RerankRespectsTopKWhenPoolSmaller(t *testing.T) {
	// Even when the reranker pool size is smaller than topK, Retrieve should
	// still fetch at least topK candidates so the result has the requested
	// length.
	p, _ := setupDensePipeline(t, 5)
	rec := &recordingReranker{}
	p.SetReranker(rec, 2) // poolSize < topK

	got, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(rec.seen) != 5 {
		t.Errorf("reranker saw %d docs, want 5 (k floor)", len(rec.seen))
	}
	if len(got.Chunks) != 5 {
		t.Errorf("result len = %d, want 5", len(got.Chunks))
	}
}

func TestRetrieve_RerankWithHybridUsesPoolDepth(t *testing.T) {
	// Hybrid retrieval already enforces a 30-candidate floor. When rerank
	// pool is bigger, the floor should grow to match — verify the reranker
	// actually sees the bigger pool.
	p, _ := setupHybridPipeline(t)
	rec := &recordingReranker{}
	p.SetReranker(rec, 50)

	_, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	// Hybrid fixture only has 3 chunks, so rec.seen will be 3, but the
	// candidate pool *attempted* was 50 (kEach=50). What we can assert is
	// that the reranker was called with everything available (3).
	if len(rec.seen) != 3 {
		t.Errorf("reranker saw %d docs, want 3 (all hybrid candidates)", len(rec.seen))
	}
}

func TestRetrieve_RerankSingleCandidateSkipsReranker(t *testing.T) {
	// With only one candidate, rerank can't change anything — verify we
	// short-circuit rather than spending a network round-trip.
	p, _ := setupDensePipeline(t, 1)
	rr := &reverseReranker{}
	p.SetReranker(rr, 1)

	if _, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{}); err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if rr.calls != 0 {
		t.Errorf("reranker should not be called with 1 candidate, got calls = %d", rr.calls)
	}
}
