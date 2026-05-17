package agent

import (
	"context"
	"strings"
	"testing"
	"time"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

func TestSemanticSearchTool_Execute_NoFilter(t *testing.T) {
	tool, s := setupSemanticSearchFixture(t)
	ctx := context.Background()
	seedDocAndCrawlPage(t, s, "doc-a", "https://example.com/a", "Alpha", "alpha-content", "alpha-source")
	seedDocAndCrawlPage(t, s, "doc-b", "https://example.com/b", "Beta", "beta-content", "beta-source")

	res, err := tool.Execute(ctx, `{"query":"content"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// With no filter, both docs should be searchable. We only assert that at
	// least one matches; ranking with the fixed embedder is irrelevant here.
	if !strings.Contains(res.Text, "alpha-content") && !strings.Contains(res.Text, "beta-content") {
		t.Errorf("unscoped search should return results, got: %s", res.Text)
	}
}

// setupSemanticSearchFixture builds an in-memory store + RAG pipeline +
// SemanticSearchTool. Mirrors setupReadDocFixture but exposes the tool
// directly for tag/scope testing.
func setupSemanticSearchFixture(t *testing.T) (*SemanticSearchTool, *store.SQLiteStore) {
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
	h := llm.NewEmbedderHandle(emb)
	r := rag.NewRetriever(s, h, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             5,
		MaxContextTokens: 4096,
	})

	tool := NewSemanticSearchTool(r)
	return tool, s
}

// seedDocAndCrawlPage creates a document AND a crawl_pages row pointing at
// it from the named source. The source_key linkage is what RAG retrieval
// reads when scoping by SourceKeys.
func seedDocAndCrawlPage(t *testing.T, s *store.SQLiteStore, id, url, title, content, sourceKey string) {
	t.Helper()
	ctx := context.Background()
	u := url
	doc := &domain.Document{
		ID:       id,
		Title:    title,
		Content:  content,
		URL:      &u,
		Language: domain.LanguageES,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("create doc %s: %v", id, err)
	}
	chunks := []domain.Chunk{{Content: content, ChunkIndex: 0, SectionTitle: ""}}
	embs := [][]float32{{1, 0, 0, 0}}
	if err := s.CreateChunks(ctx, id, chunks, embs); err != nil {
		t.Fatalf("create chunks %s: %v", id, err)
	}
	docID := id
	now := time.Now()
	if err := s.UpsertCrawlPage(ctx, &domain.CrawlPage{
		SourceKey:     sourceKey,
		URL:           url,
		ContentType:   "html",
		Title:         title,
		DocID:         &docID,
		Status:        domain.CrawlPageFetched,
		LastCrawledAt: &now,
	}); err != nil {
		t.Fatalf("upsert crawl page %s: %v", id, err)
	}
}
