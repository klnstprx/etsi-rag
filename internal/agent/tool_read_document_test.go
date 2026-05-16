package agent

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

// fixedEmbedder yields the same vector for every input — fine for the tool
// tests where ranking and embedding correctness are the rag/store layer's
// responsibility (covered there). The agent-level tests only verify routing
// and output shape.
type fixedEmbedder struct{ vec []float32 }

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

var _ llm.Embedder = fixedEmbedder{}

func setupReadDocFixture(t *testing.T) (*ReadDocumentTool, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := store.Migrate(db); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	const dim = 4
	s := store.NewSQLiteStore(db, func() int { return dim })

	emb := fixedEmbedder{vec: []float32{1, 0, 0, 0}}
	p := rag.NewPipeline(s, emb, testutil.FakeChunker{}, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             3,
		MaxContextTokens: 1024,
	})

	tool := NewReadDocumentTool(p, s)
	tool.SetDefaultTopK(3) // mirrors retrieval config TopK for the fixture
	return tool, s
}

// seedDoc inserts a document with the given URL and chunks under embeddings
// that put chunk[i] roughly along the i-th axis. The fixedEmbedder always
// queries axis-0, so chunk[0] always wins ranking.
func seedDoc(t *testing.T, s *store.SQLiteStore, id, url, title string, contents []string) {
	t.Helper()
	ctx := context.Background()
	u := url
	full := strings.Join(contents, "\n\n")
	doc := &domain.Document{
		ID:       id,
		Title:    title,
		Content:  full,
		URL:      &u,
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("create doc %s: %v", id, err)
	}
	chunks := make([]domain.Chunk, len(contents))
	embs := make([][]float32, len(contents))
	for i, c := range contents {
		chunks[i] = domain.Chunk{Content: c, ChunkIndex: i, SectionTitle: ""}
		embs[i] = []float32{0, 0, 0, 0}
		embs[i][i%4] = 1.0
	}
	if err := s.CreateChunks(ctx, id, chunks, embs); err != nil {
		t.Fatalf("create chunks %s: %v", id, err)
	}
}

func TestReadDocument_Metadata(t *testing.T) {
	tool, _ := setupReadDocFixture(t)

	if tool.Name() != "read_document" {
		t.Errorf("Name = %q, want read_document", tool.Name())
	}
	if tool.Description() == "" {
		t.Error("Description is empty")
	}
	params := tool.Parameters()
	required, ok := params["required"].([]string)
	if !ok || len(required) != 1 || required[0] != "url" {
		t.Errorf("required = %v, want [url]", params["required"])
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties missing or wrong type")
	}
	for _, name := range []string{"url", "query", "top_k"} {
		if _, ok := props[name]; !ok {
			t.Errorf("missing property %q", name)
		}
	}
}

func TestReadDocument_WholeDocumentMode(t *testing.T) {
	tool, s := setupReadDocFixture(t)
	seedDoc(t, s, "doc-curso", "https://uma.es/curso", "Curso 1",
		[]string{"Primer párrafo del documento.", "Segundo párrafo con datos."})

	got, err := tool.Execute(context.Background(),
		`{"url":"https://uma.es/curso"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Whole-doc mode returns documents.content directly.
	if !strings.Contains(got.Text, "Primer párrafo") {
		t.Errorf("text missing first paragraph: %q", got.Text)
	}
	if !strings.Contains(got.Text, "Segundo párrafo") {
		t.Errorf("text missing second paragraph: %q", got.Text)
	}
	if len(got.Citations) != 1 {
		t.Fatalf("expected 1 citation, got %d", len(got.Citations))
	}
	c := got.Citations[0]
	if c.Title != "Curso 1" || c.URL != "https://uma.es/curso" {
		t.Errorf("citation = %+v", c)
	}
}

func TestReadDocument_QueryMode(t *testing.T) {
	tool, s := setupReadDocFixture(t)
	// Three chunks; the fixedEmbedder hits axis 0 → chunk 0 ranks first.
	seedDoc(t, s, "doc-q", "https://uma.es/q", "Q-Doc",
		[]string{"chunk zero text", "chunk one text", "chunk two text"})

	got, err := tool.Execute(context.Background(),
		`{"url":"https://uma.es/q","query":"anything","top_k":2}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Query mode formats with bracketed indices like BuildContext.
	if !strings.Contains(got.Text, "[1]") {
		t.Errorf("query mode should produce [1] block; got: %q", got.Text)
	}
	if !strings.Contains(got.Text, "Q-Doc") {
		t.Errorf("query mode should include doc title; got: %q", got.Text)
	}
	// Top-2 → only first two chunks should appear.
	if !strings.Contains(got.Text, "chunk zero text") {
		t.Errorf("top-2 missing chunk zero: %q", got.Text)
	}
	if strings.Contains(got.Text, "chunk two text") {
		t.Errorf("top-2 should not include chunk 2: %q", got.Text)
	}
	if len(got.Citations) != 1 {
		t.Errorf("expected 1 citation, got %d", len(got.Citations))
	}
}

func TestReadDocument_QueryModeDefaultTopK(t *testing.T) {
	tool, s := setupReadDocFixture(t)
	seedDoc(t, s, "doc-default", "https://uma.es/d", "DefaultDoc",
		[]string{"a", "b", "c", "d", "e", "f", "g", "h"})

	got, err := tool.Execute(context.Background(),
		`{"url":"https://uma.es/d","query":"x"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Default top_k should be the pipeline's TopK (=3 in fixture).
	count := strings.Count(got.Text, "DefaultDoc")
	if count != 3 {
		t.Errorf("expected 3 results in default mode, got %d (text=%q)", count, got.Text)
	}
}

func TestReadDocument_URLMissReturnsText(t *testing.T) {
	tool, _ := setupReadDocFixture(t)

	got, err := tool.Execute(context.Background(),
		`{"url":"https://nope.example/missing"}`)
	if err != nil {
		t.Fatalf("Execute should not return Go error on URL miss, got: %v", err)
	}
	if !strings.Contains(strings.ToLower(got.Text), "not found") {
		t.Errorf("expected 'not found' in text, got: %q", got.Text)
	}
	if len(got.Citations) != 0 {
		t.Errorf("expected no citations on miss, got %d", len(got.Citations))
	}
}

func TestReadDocument_QueryNoHitsReturnsText(t *testing.T) {
	// Doc exists but has no chunks (degenerate corner case). Query mode must
	// not crash; it returns a graceful "no relevant sections" reply.
	tool, s := setupReadDocFixture(t)
	url := "https://uma.es/empty"
	doc := &domain.Document{
		ID: "empty", Title: "Empty", Content: "", URL: &url,
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(context.Background(), doc); err != nil {
		t.Fatal(err)
	}

	got, err := tool.Execute(context.Background(),
		`{"url":"https://uma.es/empty","query":"anything"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got.Text == "" {
		t.Error("expected explanatory text, got empty")
	}
}

func TestReadDocument_StripsImageMarkers(t *testing.T) {
	tool, s := setupReadDocFixture(t)
	url := "https://uma.es/m"
	doc := &domain.Document{
		ID:    "doc-mark",
		Title: "Marker Doc",
		// Whole-doc mode reads documents.content directly; markers must be
		// stripped before the LLM sees them, mirroring SemanticSearchTool.
		Content:  "Antes [[image:p1]] del párrafo y después [[image:p2]].",
		URL:      &url,
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(context.Background(), doc); err != nil {
		t.Fatal(err)
	}

	got, err := tool.Execute(context.Background(), `{"url":"https://uma.es/m"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.Contains(got.Text, "[[image:") {
		t.Errorf("text still contains markers: %q", got.Text)
	}
	if !strings.Contains(got.Text, "Antes") || !strings.Contains(got.Text, "después") {
		t.Errorf("surrounding text not preserved: %q", got.Text)
	}
}

func TestReadDocument_BadJSONIsError(t *testing.T) {
	tool, _ := setupReadDocFixture(t)
	if _, err := tool.Execute(context.Background(), `{not json`); err == nil {
		t.Fatal("expected JSON parse error")
	}
}
