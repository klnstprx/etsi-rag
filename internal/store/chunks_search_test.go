package store

import (
	"context"
	"path/filepath"
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

// FTS5 operators (AND, OR, NOT, NEAR) in the query are plain search terms,
// and quoting them must keep the tokenizer's case and diacritic folding.
func TestSearchKeyword_OperatorWordsInQuery(t *testing.T) {
	store := setupTestStore(t)
	seedChunks(t, store, "doc-1", []string{
		"Plazo de matrícula del TFG y fecha de defensa.",
		"Información general no relacionada.",
		"The deadline is NOT extendable near the end of term.",
	})

	for _, tc := range []struct {
		query     string
		wantChunk int
	}{
		{"matrícula OR plazo", 0},
		{"MATRICULA or Plazo", 0},
		{"TFG AND defensa", 0},
		{"tfg and defensa", 0},
		{"NOT", 2},
		{"not", 2},
		{"NEAR", 2},
		{"near(extendable)", 2},
		{"OR AND NOT NEAR", 2},
	} {
		results, err := store.SearchKeyword(context.Background(), tc.query, 5)
		if err != nil {
			t.Errorf("query %q: unexpected error: %v", tc.query, err)
			continue
		}
		if len(results) == 0 {
			t.Errorf("query %q: expected chunk %d, got no results", tc.query, tc.wantChunk)
			continue
		}
		if results[0].Chunk.ChunkIndex != tc.wantChunk {
			t.Errorf("query %q: top result should be chunk %d, got chunk %d: %q",
				tc.query, tc.wantChunk, results[0].Chunk.ChunkIndex, results[0].Chunk.Content)
		}
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
		Language: domain.LanguageES,
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
		Language: domain.LanguageES,
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
		Language: domain.LanguageES,
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
		Language: domain.LanguageES,
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

// TestMigrate_AddsSearchPrefixToOldSchema checks that a database created
// before chunks.search_prefix existed is upgraded in place: chunks indexed by
// the old triggers can still be deleted cleanly, and new chunks are found by
// their search prefix.
func TestMigrate_AddsSearchPrefixToOldSchema(t *testing.T) {
	db, err := OpenDB(filepath.Join(t.TempDir(), "old.db"))
	if err != nil {
		t.Fatalf("OpenDB: %v", err)
	}
	defer db.Close()
	old := `
		CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, url TEXT,
			language TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
		CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT,
			doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, content TEXT NOT NULL,
			chunk_index INTEGER NOT NULL, section_title TEXT NOT NULL DEFAULT '',
			image_refs TEXT NOT NULL DEFAULT '', UNIQUE(doc_id, chunk_index));
		CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='chunks', content_rowid='id',
			tokenize='unicode61 remove_diacritics 2');
		CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
			INSERT INTO chunks_fts(rowid, content) VALUES (new.id, coalesce(new.section_title, '') || ' ' || new.content);
		END;
		CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
			INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, coalesce(old.section_title, '') || ' ' || old.content);
		END;
		INSERT INTO documents (id, title, content, language) VALUES ('old', 'Old', 'x', 'es');
		INSERT INTO chunks (doc_id, content, chunk_index, section_title) VALUES ('old', 'reglamento antiguo', 0, 'Normativa');`
	if _, err := db.Exec(old); err != nil {
		t.Fatalf("creating old schema: %v", err)
	}
	if err := Migrate(db); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	s := NewSQLiteStore(db, func() int { return 2 })
	ctx := context.Background()

	if err := s.DeleteDocument(ctx, "old"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')`); err != nil {
		t.Fatalf("FTS index inconsistent after deleting an old-trigger row: %v", err)
	}

	doc := &domain.Document{ID: "new", Title: "Grado en Ingeniería Informática", Content: "x", Language: domain.LanguageES}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("CreateDocument: %v", err)
	}
	chunk := domain.Chunk{Content: "Plazas nuevo ingreso 150", SearchPrefix: doc.Title}
	if err := s.CreateChunks(ctx, "new", []domain.Chunk{chunk}, [][]float32{{1, 0}}); err != nil {
		t.Fatalf("CreateChunks: %v", err)
	}
	got, err := s.SearchKeyword(ctx, "informatica", 5)
	if err != nil || len(got) != 1 || got[0].Chunk.DocID != "new" {
		t.Fatalf("SearchKeyword by prefix word = %v, %v; want the new chunk", got, err)
	}
}
