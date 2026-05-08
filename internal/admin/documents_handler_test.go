package admin

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"etsi-rag/internal/domain"
)

func TestResolveSourceFilter(t *testing.T) {
	all := []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", Tags: []string{"degree-cs"}},
		{Key: "beta", Name: "Beta", Tags: []string{"degree-cs", "mobility"}},
		{Key: "gamma", Name: "Gamma", Tags: []string{"library"}},
	}

	tests := []struct {
		name      string
		source    string
		tag       string
		wantKeys  []string
		wantEmpty bool
		anyOrder  bool // when keys returned by tag lookup, comparison is set-based
	}{
		{"no filters", "", "", nil, false, false},
		{"explicit source", "alpha", "", []string{"alpha"}, false, false},
		{"unknown source still passes through", "ghost", "", []string{"ghost"}, false, false},
		{"single tag → one key", "", "library", []string{"gamma"}, false, false},
		{"single tag → many keys", "", "degree-cs", []string{"alpha", "beta"}, false, true},
		{"unknown tag forces empty", "", "no-such-tag", nil, true, false},
		{"source + tag intersect", "beta", "degree-cs", []string{"beta"}, false, false},
		{"source + tag mismatch forces empty", "alpha", "library", nil, true, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotKeys, gotEmpty := resolveSourceFilter(all, tt.source, tt.tag)
			if gotEmpty != tt.wantEmpty {
				t.Errorf("forceEmpty = %v, want %v", gotEmpty, tt.wantEmpty)
			}
			if tt.anyOrder {
				sort.Strings(gotKeys)
				sort.Strings(tt.wantKeys)
			}
			if !reflect.DeepEqual(gotKeys, tt.wantKeys) {
				t.Errorf("keys = %v, want %v", gotKeys, tt.wantKeys)
			}
		})
	}
}

func TestCollectTags_SortedAndDeduped(t *testing.T) {
	got := collectTags([]domain.CrawlSource{
		{Tags: []string{"b", "a"}},
		{Tags: []string{"a", "c", ""}},
		{Tags: nil},
	})
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("collectTags = %v, want %v", got, want)
	}
}

func TestDocumentList_RendersSourceAndTagsForDoc(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha Source", SeedURL: "https://example.com/a/", Enabled: true, Tags: []string{"degree-cs", "mobility"}},
	})
	ctx := context.Background()

	if err := h.store.CreateDocument(ctx, &domain.Document{
		ID: "doc-with-source", Title: "Linked", Content: "content", Language: domain.LanguageES,
	}); err != nil {
		t.Fatal(err)
	}
	docID := "doc-with-source"
	now := time.Now()
	if err := h.store.UpsertCrawlPage(ctx, &domain.CrawlPage{
		SourceKey:     "alpha",
		URL:           "https://example.com/a/page",
		ContentType:   "html",
		Title:         "Linked",
		DocID:         &docID,
		Status:        domain.CrawlPageFetched,
		LastCrawledAt: &now,
	}); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodGet, "/admin/documents", nil)
	w := httptest.NewRecorder()
	h.DocumentList(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()

	// The source's display name and both inherited tags must appear.
	if !strings.Contains(body, "Alpha Source") {
		t.Errorf("body missing source display name; got:\n%s", body)
	}
	for _, tag := range []string{"degree-cs", "mobility"} {
		if !strings.Contains(body, tag) {
			t.Errorf("body missing tag %q; got:\n%s", tag, body)
		}
	}
}

func TestDocumentList_FiltersBySource(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", SeedURL: "https://example.com/a/", Enabled: true},
		{Key: "beta", Name: "Beta", SeedURL: "https://example.com/b/", Enabled: true},
	})
	seedDocWithSource(t, h, "doc-alpha", "Doc Alpha", "alpha", "https://example.com/a/p1")
	seedDocWithSource(t, h, "doc-beta", "Doc Beta", "beta", "https://example.com/b/p1")

	req := httptest.NewRequest(http.MethodGet, "/admin/documents?source=alpha", nil)
	w := httptest.NewRecorder()
	h.DocumentList(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Doc Alpha") {
		t.Errorf("body missing alpha doc; got:\n%s", body)
	}
	if strings.Contains(body, "Doc Beta") {
		t.Errorf("body should not contain beta doc when filtering source=alpha; got:\n%s", body)
	}
}

func TestDocumentList_FiltersByTag(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", SeedURL: "https://example.com/a/", Enabled: true, Tags: []string{"degree-cs"}},
		{Key: "beta", Name: "Beta", SeedURL: "https://example.com/b/", Enabled: true, Tags: []string{"library"}},
	})
	seedDocWithSource(t, h, "doc-alpha", "Doc Alpha", "alpha", "https://example.com/a/p1")
	seedDocWithSource(t, h, "doc-beta", "Doc Beta", "beta", "https://example.com/b/p1")

	req := httptest.NewRequest(http.MethodGet, "/admin/documents?tag=degree-cs", nil)
	w := httptest.NewRecorder()
	h.DocumentList(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Doc Alpha") {
		t.Errorf("tag filter should include the alpha doc; got:\n%s", body)
	}
	if strings.Contains(body, "Doc Beta") {
		t.Errorf("tag filter should exclude beta doc; got:\n%s", body)
	}
}

func TestDocumentList_UnknownTagShowsEmpty(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", SeedURL: "https://example.com/a/", Enabled: true, Tags: []string{"degree-cs"}},
	})
	seedDocWithSource(t, h, "doc-alpha", "Doc Alpha", "alpha", "https://example.com/a/p1")

	req := httptest.NewRequest(http.MethodGet, "/admin/documents?tag=ghost-tag", nil)
	w := httptest.NewRecorder()
	h.DocumentList(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if strings.Contains(body, "Doc Alpha") {
		t.Errorf("body should not include any doc when tag has no sources; got:\n%s", body)
	}
}

func TestDocumentDetail_ShowsSourceAndTags(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha Source", SeedURL: "https://example.com/a/", Enabled: true, Tags: []string{"degree-cs"}},
	})
	seedDocWithSource(t, h, "doc-detail", "Detail Doc", "alpha", "https://example.com/a/p1")

	req := httptest.NewRequest(http.MethodGet, "/admin/documents/doc-detail", nil)
	req = withChiParam(req, "id", "doc-detail")
	w := httptest.NewRecorder()
	h.DocumentDetail(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Alpha Source") {
		t.Errorf("detail body missing source name; got:\n%s", body)
	}
	if !strings.Contains(body, "degree-cs") {
		t.Errorf("detail body missing tag; got:\n%s", body)
	}
}

// TestDocumentDelete_PreservesFiltersFromHXCurrentURL verifies that after a
// delete the post-action re-render keeps the active source/tag filter from
// the HTMX HX-Current-URL header. We seed two docs in two sources, delete
// the alpha doc with HX-Current-URL=...?source=alpha, and assert beta is
// not in the rendered fragment (it would be present if filters were lost).
func TestDocumentDelete_PreservesFiltersFromHXCurrentURL(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", SeedURL: "https://example.com/a/", Enabled: true},
		{Key: "beta", Name: "Beta Distinctive Name", SeedURL: "https://example.com/b/", Enabled: true},
	})
	seedDocWithSource(t, h, "doc-alpha", "Doc Alpha", "alpha", "https://example.com/a/p1")
	seedDocWithSource(t, h, "doc-beta", "Doc Beta Distinctive Name", "beta", "https://example.com/b/p1")

	req := httptest.NewRequest(http.MethodDelete, "/admin/documents/doc-alpha", nil)
	req = withChiParam(req, "id", "doc-alpha")
	req.Header.Set("HX-Request", "true")
	req.Header.Set("HX-Current-URL", "http://localhost/admin/documents?source=alpha")
	w := httptest.NewRecorder()
	h.DocumentDelete(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if strings.Contains(body, "Doc Alpha") {
		t.Errorf("alpha doc should be deleted; body still contains it:\n%s", body)
	}
	if strings.Contains(body, "Doc Beta Distinctive Name") {
		t.Errorf("beta doc should be excluded by source=alpha filter, but appeared in body:\n%s", body)
	}
}

// TestDocumentDelete_NoHXCurrentURLFallsBackToUnfiltered verifies the
// fallback path: when HX-Current-URL is absent (e.g. a non-HTMX caller), the
// post-action render shows the unfiltered list. Otherwise we'd lose the
// remaining doc and the operator would think the delete cascaded.
func TestDocumentDelete_NoHXCurrentURLFallsBackToUnfiltered(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "alpha", Name: "Alpha", SeedURL: "https://example.com/a/", Enabled: true},
		{Key: "beta", Name: "Beta", SeedURL: "https://example.com/b/", Enabled: true},
	})
	seedDocWithSource(t, h, "doc-alpha", "Doc Alpha", "alpha", "https://example.com/a/p1")
	seedDocWithSource(t, h, "doc-beta", "Doc Beta Survivor", "beta", "https://example.com/b/p1")

	req := httptest.NewRequest(http.MethodDelete, "/admin/documents/doc-alpha", nil)
	req = withChiParam(req, "id", "doc-alpha")
	// Deliberately no HX-Current-URL header.
	w := httptest.NewRecorder()
	h.DocumentDelete(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "Doc Beta Survivor") {
		t.Errorf("beta doc should still appear in unfiltered fallback render; body:\n%s", body)
	}
}

// seedDocWithSource creates a document and the corresponding crawl_pages row
// linking it to the given source. Reduces boilerplate in the filter tests.
func seedDocWithSource(t *testing.T, h *Handler, docID, title, sourceKey, url string) {
	t.Helper()
	ctx := context.Background()
	if err := h.store.CreateDocument(ctx, &domain.Document{
		ID: docID, Title: title, Content: "content", Language: domain.LanguageES,
	}); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	id := docID
	if err := h.store.UpsertCrawlPage(ctx, &domain.CrawlPage{
		SourceKey:     sourceKey,
		URL:           url,
		ContentType:   "html",
		Title:         title,
		DocID:         &id,
		Status:        domain.CrawlPageFetched,
		LastCrawledAt: &now,
	}); err != nil {
		t.Fatal(err)
	}
}
