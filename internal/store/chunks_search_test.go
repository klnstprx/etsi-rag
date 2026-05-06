package store

import (
	"context"
	"testing"

	"etsi-rag/internal/domain"
)

// seedChunks creates a parent document with the given chunks, all in Spanish.
// Embeddings are dummy zero-vectors of testEmbeddingDimension — BM25 doesn't
// touch them, but CreateChunks requires the dimension to match.
func seedChunks(t *testing.T, store *SQLiteStore, docID string, contents []string) {
	t.Helper()
	ctx := context.Background()

	doc := &domain.Document{
		ID:       docID,
		Title:    "Test " + docID,
		Content:  "joined content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document %s: %v", docID, err)
	}

	chunks := make([]domain.Chunk, len(contents))
	embeddings := make([][]float32, len(contents))
	for i, content := range contents {
		chunks[i] = domain.Chunk{Content: content, ChunkIndex: i}
		embeddings[i] = make([]float32, testEmbeddingDimension)
		embeddings[i][i%testEmbeddingDimension] = 1.0
	}
	if err := store.CreateChunks(ctx, docID, chunks, embeddings); err != nil {
		t.Fatalf("creating chunks for %s: %v", docID, err)
	}
}

func TestSearchKeyword_EmptyCorpus(t *testing.T) {
	store := setupTestStore(t)
	results, err := store.SearchKeyword(context.Background(), "anything", 5)
	if err != nil {
		t.Fatalf("SearchKeyword: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("empty corpus: expected 0 results, got %d", len(results))
	}
}

func TestSearchKeyword_EmptyQueryReturnsEmpty(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{"some content"})

	for _, q := range []string{"", "   ", "\t\n"} {
		results, err := store.SearchKeyword(context.Background(), q, 5)
		if err != nil {
			t.Errorf("query %q: unexpected error: %v", q, err)
		}
		if len(results) != 0 {
			t.Errorf("query %q: expected 0 results, got %d", q, len(results))
		}
	}
}

func TestSearchKeyword_ExactTokenMatch(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{
		"Algo sin nada interesante.",
		"Número de Plazas (Memoria de Verificación): 140 — Lenguas: Castellano, Inglés.",
		"Otro chunk irrelevante con palabras genéricas.",
	})

	results, err := store.SearchKeyword(context.Background(), "plazas 140", 5)
	if err != nil {
		t.Fatalf("SearchKeyword: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one match for 'plazas 140'")
	}
	// The Plazas chunk is index 1 in the seed list.
	if results[0].Chunk.ChunkIndex != 1 {
		t.Errorf("top result should be chunk 1 (Plazas), got chunk %d: %q",
			results[0].Chunk.ChunkIndex, results[0].Chunk.Content)
	}
	if results[0].Score <= 0 {
		t.Errorf("top result should have positive score (negated bm25), got %f", results[0].Score)
	}
}

func TestSearchKeyword_DiacriticFolding(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{
		"Coordinadora: María José Cisneros Portillo.",
		"Otros nombres: Juan, Pedro, Ana.",
	})

	// Query without accent should still match a chunk that has accents.
	results, err := store.SearchKeyword(context.Background(), "maria", 5)
	if err != nil {
		t.Fatalf("SearchKeyword: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected diacritic-folded match: 'maria' should hit 'María'")
	}
	if results[0].Chunk.ChunkIndex != 0 {
		t.Errorf("top result should be the María chunk, got chunk %d: %q",
			results[0].Chunk.ChunkIndex, results[0].Chunk.Content)
	}
}

func TestSearchKeyword_SpecialCharsInQuerySurvive(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{
		"Contacto: secretariaetsii@uma.es teléfono 952 13 14 95.",
		"Información general no relacionada.",
	})

	// Query contains characters that are FTS5 syntax (?, @, .) — must not crash.
	for _, q := range []string{
		"¿qué? secretariaetsii@uma.es",
		`"teléfono" (952)`,
		"a*b:c^d",
	} {
		results, err := store.SearchKeyword(context.Background(), q, 5)
		if err != nil {
			t.Errorf("query %q: unexpected error: %v", q, err)
			continue
		}
		// We don't assert what was returned (the sanitiser may strip everything),
		// only that the call succeeded without surfacing an FTS5 syntax error.
		_ = results
	}
}

func TestSearchKeyword_TopKLimit(t *testing.T) {
	store := setupTestStore(t)
	contents := make([]string, 10)
	for i := range contents {
		contents[i] = "El término común aparece en todos los chunks."
	}
	seedChunks(t, store, "doc-1", contents)

	results, err := store.SearchKeyword(context.Background(), "término común", 3)
	if err != nil {
		t.Fatalf("SearchKeyword: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("topK=3 over 10 matching chunks: expected 3 results, got %d", len(results))
	}
}

func TestSearchKeyword_DocumentDeleteRemovesFromFTS(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()
	seedChunks(t, store, "doc-1", []string{"unique-token-xyz appears here only."})

	// Sanity: the unique token finds the chunk.
	results, err := store.SearchKeyword(ctx, "unique-token-xyz", 5)
	if err != nil {
		t.Fatalf("SearchKeyword pre-delete: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("pre-delete: expected to find the unique token")
	}

	// Deleting the parent document cascades to chunks; the AFTER DELETE
	// trigger on chunks must propagate to chunks_fts.
	if err := store.DeleteDocument(ctx, "doc-1"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}

	results, err = store.SearchKeyword(ctx, "unique-token-xyz", 5)
	if err != nil {
		t.Fatalf("SearchKeyword post-delete: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("post-delete: expected 0 results, got %d (FTS5 trigger not firing on cascade?)", len(results))
	}
}

// SearchInDocument is the doc-scoped dense search powering read_document's
// query mode. It mirrors SearchSimilar but filters chunks to a single
// document — small enough that we don't bother with the embedding cache.

func TestSearchInDocument_RanksByCosine(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	// Two docs, three chunks each, distinct embeddings.
	doc1 := &domain.Document{
		ID: "doc-1", Title: "Doc 1", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc1); err != nil {
		t.Fatal(err)
	}
	chunks := []domain.Chunk{
		{Content: "axis-x", ChunkIndex: 0},
		{Content: "axis-y", ChunkIndex: 1},
		{Content: "diagonal", ChunkIndex: 2},
	}
	embeddings := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.7, 0.7, 0.0, 0.0},
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	// A second doc with chunks that, by themselves, would dominate the global
	// top-K. SearchInDocument must ignore them.
	doc2 := &domain.Document{
		ID: "doc-2", Title: "Doc 2", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc2); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-2",
		[]domain.Chunk{{Content: "perfect-x", ChunkIndex: 0}},
		[][]float32{{1.0, 0.0, 0.0, 0.0}},
	); err != nil {
		t.Fatal(err)
	}

	query := []float32{1.0, 0.0, 0.0, 0.0}
	results, err := s.SearchInDocument(ctx, "doc-1", query, 5)
	if err != nil {
		t.Fatalf("SearchInDocument: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results (all of doc-1), got %d", len(results))
	}
	// All results must be from doc-1.
	for i, r := range results {
		if r.Chunk.DocID != "doc-1" {
			t.Errorf("result[%d] from foreign doc %q", i, r.Chunk.DocID)
		}
	}
	// Top-1 is the axis-x chunk (perfect cosine match).
	if results[0].Chunk.ChunkIndex != 0 {
		t.Errorf("top result should be chunk 0 (axis-x), got chunk %d: %q",
			results[0].Chunk.ChunkIndex, results[0].Chunk.Content)
	}
	// Strictly descending scores.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("not score-descending at i=%d: %f > %f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}

func TestSearchInDocument_TopKLimit(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-1", Title: "Doc", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	chunks := make([]domain.Chunk, 5)
	embs := make([][]float32, 5)
	for i := range chunks {
		chunks[i] = domain.Chunk{Content: "c", ChunkIndex: i}
		embs[i] = []float32{float32(i+1) * 0.1, 0, 0, 0}
	}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embs); err != nil {
		t.Fatal(err)
	}

	query := []float32{1, 0, 0, 0}

	t.Run("topK less than total", func(t *testing.T) {
		got, err := s.SearchInDocument(ctx, "doc-1", query, 2)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 {
			t.Errorf("got %d, want 2", len(got))
		}
	})

	t.Run("topK greater than total returns all", func(t *testing.T) {
		got, err := s.SearchInDocument(ctx, "doc-1", query, 100)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 5 {
			t.Errorf("got %d, want 5 (all chunks)", len(got))
		}
	})

	t.Run("topK zero returns empty", func(t *testing.T) {
		got, err := s.SearchInDocument(ctx, "doc-1", query, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Errorf("got %d, want 0", len(got))
		}
	})
}

func TestSearchInDocument_UnknownDocReturnsEmpty(t *testing.T) {
	// Mirror SearchSimilar's "no results = empty slice, not error" semantics.
	// The agent tool maps this to a graceful "document not found / no hits"
	// text reply rather than propagating an error.
	s := setupTestStore(t)
	got, err := s.SearchInDocument(context.Background(), "missing-doc",
		[]float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("unexpected error for unknown doc: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected 0 results, got %d", len(got))
	}
}

func TestSearchInDocument_EmbeddingDimMismatch(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID: "doc-1", Title: "Doc", Content: "c",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-1",
		[]domain.Chunk{{Content: "c", ChunkIndex: 0}},
		[][]float32{{1, 0, 0, 0}},
	); err != nil {
		t.Fatal(err)
	}

	// Wrong-length query → error, not silent zero score.
	_, err := s.SearchInDocument(ctx, "doc-1", []float32{1, 0}, 5)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
}

func TestSearchKeyword_OrdersByRelevance(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{
		"plazas plazas plazas plazas — chunk dense en la palabra 'plazas'.",
		"Una sola mención: plazas.",
		"Sin la palabra clave.",
	})

	results, err := store.SearchKeyword(context.Background(), "plazas", 5)
	if err != nil {
		t.Fatalf("SearchKeyword: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("expected at least 2 matches, got %d", len(results))
	}
	// BM25 should rank the dense-mention chunk above the single-mention one.
	if results[0].Chunk.ChunkIndex != 0 {
		t.Errorf("dense-match chunk should rank first, got chunk %d: %q",
			results[0].Chunk.ChunkIndex, results[0].Chunk.Content)
	}
	// And scores should be strictly descending (higher = better).
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not in score-descending order at i=%d: %f > %f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}
