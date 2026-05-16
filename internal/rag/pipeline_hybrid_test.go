package rag

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// fixedEmbedder always returns the same query vector. Useful for tests where
// we want to control the ranking entirely through chunk embeddings, without
// the embedder rewriting the query side.
type fixedEmbedder struct {
	vec []float32
}

func (f fixedEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return f.vec, nil
}

func (f fixedEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = f.vec
	}
	return out, nil
}

// setupHybridPipeline constructs a Pipeline configured for hybrid retrieval,
// seeded with three chunks whose dense ranking is fixed by hand-rolled
// embeddings. Content is crafted so a unique lexical token appears only in
// the dense-worst chunk, exposing whether RRF pulls it up.
func setupHybridPipeline(t *testing.T) (*Pipeline, *store.SQLiteStore) {
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
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("create doc: %v", err)
	}

	// Query embedder always returns [0.5, 0.5, 0, 0].
	// Dense cosine ranking against that:
	//   A — [0.5, 0.5, 0, 0] → 1.0    (rank 1 dense)
	//   B — [0.7, 0.3, 0, 0] → ~0.949 (rank 2 dense)
	//   C — [0, 0, 1, 0]     → 0      (rank 3 dense, would never surface in top-1)
	chunks := []domain.Chunk{
		{Content: "información sobre temas generales sin tokens raros aquí", ChunkIndex: 0},
		{Content: "otro fragmento de contenido genérico", ChunkIndex: 1},
		{Content: "el único chunk con uniquetokenzigzag escrito explícitamente", ChunkIndex: 2},
	}
	embeddings := [][]float32{
		{0.5, 0.5, 0, 0},
		{0.7, 0.3, 0, 0},
		{0, 0, 1, 0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatalf("create chunks: %v", err)
	}

	emb := fixedEmbedder{vec: []float32{0.5, 0.5, 0, 0}}

	p := NewPipeline(s, emb, testutil.FakeChunker{}, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             3,
		MaxContextTokens: 1024,
		RetrievalMode:    "hybrid",
	})
	return p, s
}

func TestRetrieve_HybridSurfacesDenseLowChunkViaBM25(t *testing.T) {
	p, _ := setupHybridPipeline(t)
	ctx := context.Background()

	// "uniquetokenzigzag" appears only in chunk C (chunk index 2). C ranks
	// last in dense (cos=0 vs ~1.0 for A) but rank 1 in BM25. RRF should
	// pull it to the top of the merged result.
	q := "uniquetokenzigzag"
	got, err := p.Retrieve(ctx, q, RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(got.Chunks) == 0 {
		t.Fatal("hybrid retrieve returned no chunks")
	}
	if got.Chunks[0].Chunk.ChunkIndex != 2 {
		var ranks []int
		for _, c := range got.Chunks {
			ranks = append(ranks, c.Chunk.ChunkIndex)
		}
		t.Errorf("hybrid should rank chunk 2 (BM25 hit) first; got order: %v", ranks)
	}
}

func TestRetrieve_DenseModeIgnoresLexicalMatch(t *testing.T) {
	// Same fixture, but with retrieval_mode=dense — chunk 2 should NOT be
	// the top result because dense cosine ignores token overlap.
	p, _ := setupHybridPipeline(t)
	cfg := p.getConfig()
	cfg.RetrievalMode = "dense"
	p.SetConfig(cfg)

	ctx := context.Background()
	got, err := p.Retrieve(ctx, "uniquetokenzigzag", RetrieveOpts{})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(got.Chunks) == 0 {
		t.Fatal("dense retrieve returned no chunks")
	}
	if got.Chunks[0].Chunk.ChunkIndex == 2 {
		t.Errorf("dense-only should NOT rank chunk 2 first (its embedding is orthogonal); "+
			"got top chunk %d", got.Chunks[0].Chunk.ChunkIndex)
	}
}

func TestRetrieve_UnsupportedModeErrors(t *testing.T) {
	p, _ := setupHybridPipeline(t)
	cfg := p.getConfig()
	cfg.RetrievalMode = "wat"
	p.SetConfig(cfg)

	_, err := p.Retrieve(context.Background(), "anything", RetrieveOpts{})
	if err == nil {
		t.Fatal("expected error for unsupported retrieval_mode")
	}
	if !strings.Contains(err.Error(), "wat") {
		t.Errorf("error should mention bad mode value, got: %v", err)
	}
}
