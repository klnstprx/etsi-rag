package agent

import (
	"context"
	"sort"
	"strings"
	"testing"
	"time"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
	"etsi-rag/internal/testutil"
)

func TestResolveScope(t *testing.T) {
	all := []domain.CrawlSource{
		{Key: "alpha", Tags: []string{"degree-cs"}},
		{Key: "beta", Tags: []string{"degree-cs", "mobility"}},
		{Key: "gamma", Tags: []string{"library"}},
		{Key: "delta", Tags: nil},
	}
	tool := &SemanticSearchTool{Snapshot: func() []domain.CrawlSource { return all }}

	tests := []struct {
		name        string
		sourceKeys  []string
		tags        []string
		wantOK      bool
		wantKeysSet []string // sorted comparison; nil = no filter
	}{
		{"neither set → no filter", nil, nil, true, nil},
		{"explicit source_keys passes through", []string{"alpha"}, nil, true, []string{"alpha"}},
		{"single tag → one source", nil, []string{"library"}, true, []string{"gamma"}},
		{"single tag → multiple sources (OR within tag list)", nil, []string{"degree-cs"}, true, []string{"alpha", "beta"}},
		{"multiple tags compose with OR", nil, []string{"library", "mobility"}, true, []string{"beta", "gamma"}},
		{"unknown tag with no source_keys → empty (force fail)", nil, []string{"ghost-tag"}, false, nil},
		{"unknown tag but source_keys set → just the explicit keys", []string{"alpha"}, []string{"ghost-tag"}, true, []string{"alpha"}},
		{"tags + source_keys union", []string{"delta"}, []string{"library"}, true, []string{"delta", "gamma"}},
		{"dedupe overlap between source_keys and tag-resolved keys", []string{"gamma"}, []string{"library"}, true, []string{"gamma"}},
		{"empty-string entries are skipped", []string{"", "alpha"}, []string{""}, true, []string{"alpha"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotKeys, gotOK := tool.resolveScope(tt.sourceKeys, tt.tags)
			if gotOK != tt.wantOK {
				t.Fatalf("ok = %v, want %v (keys=%v)", gotOK, tt.wantOK, gotKeys)
			}
			if tt.wantKeysSet == nil {
				if gotKeys != nil {
					t.Errorf("keys = %v, want nil", gotKeys)
				}
				return
			}
			sort.Strings(gotKeys)
			want := append([]string(nil), tt.wantKeysSet...)
			sort.Strings(want)
			if len(gotKeys) != len(want) {
				t.Fatalf("keys = %v, want %v", gotKeys, want)
			}
			for i := range gotKeys {
				if gotKeys[i] != want[i] {
					t.Fatalf("keys = %v, want %v", gotKeys, want)
				}
			}
		})
	}
}

func TestResolveScope_NilSnapshot(t *testing.T) {
	tool := &SemanticSearchTool{Snapshot: nil}

	// tags-only with no snapshot → unresolvable; ok=false so the caller
	// emits the actionable-error message.
	if _, ok := tool.resolveScope(nil, []string{"mobility"}); ok {
		t.Error("tags + nil snapshot should report ok=false")
	}
	// source_keys still work without a snapshot.
	keys, ok := tool.resolveScope([]string{"alpha"}, nil)
	if !ok || len(keys) != 1 || keys[0] != "alpha" {
		t.Errorf("source_keys + nil snapshot: keys=%v ok=%v", keys, ok)
	}
}

func TestEmptyScopeMessage(t *testing.T) {
	cases := []struct {
		name        string
		sourceKeys  []string
		tags        []string
		mustContain []string
	}{
		{"tags only", nil, []string{"foo"}, []string{"tags=[foo]", "list_sources"}},
		{"source_keys only", []string{"bar"}, nil, []string{"source_keys=[bar]", "list_sources"}},
		{"both", []string{"bar"}, []string{"foo"}, []string{"tags=[foo]", "source_keys=[bar]"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := emptyScopeMessage(tc.sourceKeys, tc.tags)
			for _, frag := range tc.mustContain {
				if !strings.Contains(got, frag) {
					t.Errorf("message missing %q\nfull: %s", frag, got)
				}
			}
		})
	}
}

// --- Execute() integration with a real pipeline + in-memory store ---
//
// The fixedEmbedder + small-DB fixture is borrowed from tool_read_document_test.go;
// reusing the same shape keeps the agent test surface coherent.

func TestSemanticSearchTool_Execute_TagsResolveAndScope(t *testing.T) {
	tool, s := setupSemanticSearchFixture(t)
	ctx := context.Background()

	// Two docs. Each is associated to a separate source via crawl_pages
	// (which is how rag retrieval scopes via SourceKeys: the predicate is
	// EXISTS over crawl_pages). The fixedEmbedder always returns axis-0,
	// so chunk-0 of both docs ranks; the source filter is what differentiates.
	seedDocAndCrawlPage(t, s, "doc-a", "https://example.com/a", "Alpha title", "alpha-content alpha-token", "alpha-source")
	seedDocAndCrawlPage(t, s, "doc-b", "https://example.com/b", "Beta title", "beta-content beta-token", "beta-source")

	tool.Snapshot = func() []domain.CrawlSource {
		return []domain.CrawlSource{
			{Key: "alpha-source", Tags: []string{"degree-cs"}, Enabled: true},
			{Key: "beta-source", Tags: []string{"library"}, Enabled: true},
		}
	}

	// tag=degree-cs should scope to alpha-source only.
	res, err := tool.Execute(ctx, `{"query":"alpha","tags":["degree-cs"]}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(res.Text, "alpha-content") {
		t.Errorf("expected alpha content in result, got:\n%s", res.Text)
	}
	if strings.Contains(res.Text, "beta-content") {
		t.Errorf("beta content should be excluded by tag filter, got:\n%s", res.Text)
	}
}

func TestSemanticSearchTool_Execute_TagsAndSourceKeysUnion(t *testing.T) {
	tool, s := setupSemanticSearchFixture(t)
	ctx := context.Background()

	seedDocAndCrawlPage(t, s, "doc-a", "https://example.com/a", "Alpha", "alpha-content", "alpha-source")
	seedDocAndCrawlPage(t, s, "doc-b", "https://example.com/b", "Beta", "beta-content", "beta-source")
	seedDocAndCrawlPage(t, s, "doc-c", "https://example.com/c", "Gamma", "gamma-content", "gamma-source")

	tool.Snapshot = func() []domain.CrawlSource {
		return []domain.CrawlSource{
			{Key: "alpha-source", Tags: []string{"degree-cs"}, Enabled: true},
			{Key: "beta-source", Tags: []string{"library"}, Enabled: true},
			{Key: "gamma-source", Tags: nil, Enabled: true},
		}
	}

	// source_keys=['gamma-source'] ∪ tags=['degree-cs'] = {alpha-source, gamma-source}.
	res, err := tool.Execute(ctx, `{"query":"content","source_keys":["gamma-source"],"tags":["degree-cs"]}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(res.Text, "alpha-content") || !strings.Contains(res.Text, "gamma-content") {
		t.Errorf("expected union to surface alpha + gamma, got:\n%s", res.Text)
	}
	if strings.Contains(res.Text, "beta-content") {
		t.Errorf("beta should be excluded by union filter, got:\n%s", res.Text)
	}
}

func TestSemanticSearchTool_Execute_UnknownTagReturnsActionableError(t *testing.T) {
	tool, s := setupSemanticSearchFixture(t)
	ctx := context.Background()
	seedDocAndCrawlPage(t, s, "doc-a", "https://example.com/a", "Alpha", "alpha-content", "alpha-source")
	tool.Snapshot = func() []domain.CrawlSource {
		return []domain.CrawlSource{{Key: "alpha-source", Tags: []string{"degree-cs"}, Enabled: true}}
	}

	res, err := tool.Execute(ctx, `{"query":"alpha","tags":["nonexistent-tag"]}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(res.Text, "nonexistent-tag") {
		t.Errorf("error message should name the offending tag, got: %s", res.Text)
	}
	if !strings.Contains(res.Text, "list_sources") {
		t.Errorf("error message should point at list_sources, got: %s", res.Text)
	}
	if strings.Contains(res.Text, "alpha-content") {
		t.Errorf("must not silently fall back to a full-corpus search, got: %s", res.Text)
	}
}

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

	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{MaxChunkSize: 200, ChunkOverlap: 0})
	emb := fixedEmbedder{vec: []float32{1, 0, 0, 0}}
	p := rag.NewPipeline(s, emb, chunker, &testutil.MockCounter{}, config.RetrievalConfig{
		TopK:             5,
		MaxContextTokens: 4096,
	})

	tool := NewSemanticSearchTool(p)
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
