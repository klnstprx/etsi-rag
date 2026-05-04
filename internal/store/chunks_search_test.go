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
