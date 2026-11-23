package admin

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/sources"
)

// setupTestHandlerWithSources extends setupTestAdminHandler with a sources.toml
// seeded in a temp dir and a wired sources.Registry, matching server.New.
func setupTestHandlerWithSources(t *testing.T, seed []domain.CrawlSource) *Handler {
	t.Helper()
	h, _ := setupTestAdminHandler(t)

	path := config.SourcesPath(t.TempDir())
	if err := config.WriteSources(seed, path); err != nil {
		t.Fatalf("seeding sources.toml: %v", err)
	}
	reg := sources.New(path)
	if err := reg.Load(); err != nil {
		t.Fatalf("loading test registry: %v", err)
	}
	h.sources = reg
	return h
}

func TestSourceUpdate_RejectsInvalidKey(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "valid", Name: "Valid", SeedURL: "https://example.com/", Enabled: true},
	})

	form := url.Values{}
	form.Set("name", "x")
	form.Set("seed_url", "https://example.com/")
	form.Set("enabled", "true")

	cases := []struct {
		name string
		key  string
	}{
		{"whitespace", "   "},
		{"path separator", "foo/bar"},
		{"control char", "foo\x00bar"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Use a clean placeholder path; the handler reads the key from the
			// chi route context, not from r.URL — so injecting via withChiParam
			// exercises the validator without tripping httptest.NewRequest's
			// URL parse step on whitespace/control characters.
			req := putForm("/admin/crawler/sources/_", form)
			req = withChiParam(req, "key", tc.key)
			rec := httptest.NewRecorder()
			h.SourceUpdate(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d (body=%q)", rec.Code, http.StatusBadRequest, rec.Body.String())
			}
		})
	}
}

func TestSourceDelete_RejectsInvalidKey(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "valid", Name: "Valid", SeedURL: "https://example.com/", Enabled: true},
	})

	cases := []struct {
		name string
		key  string
	}{
		{"whitespace", "   "},
		{"path separator", "foo/bar"},
		{"control char", "foo\x00bar"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodDelete, "/admin/crawler/sources/_", nil)
			req = withChiParam(req, "key", tc.key)
			rec := httptest.NewRecorder()
			h.SourceDelete(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d (body=%q)", rec.Code, http.StatusBadRequest, rec.Body.String())
			}
		})
	}
}

func TestSourceUpdate_HappyPath(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "valid", Name: "Original", SeedURL: "https://example.com/", Enabled: true},
	})

	form := url.Values{}
	form.Set("name", "Renamed")
	form.Set("seed_url", "https://example.com/new/")
	form.Set("description", "updated")
	form.Set("tags", "a, b")
	form.Set("enabled", "true")

	req := putForm("/admin/crawler/sources/valid", form)
	req = withChiParam(req, "key", "valid")
	rec := httptest.NewRecorder()
	h.SourceUpdate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (body=%q)", rec.Code, http.StatusOK, rec.Body.String())
	}

	got, err := config.LoadSourcesFile(h.sources.Path())
	if err != nil {
		t.Fatalf("LoadSourcesFile: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len(sources) = %d, want 1", len(got))
	}
	if got[0].Name != "Renamed" || got[0].SeedURL != "https://example.com/new/" {
		t.Errorf("sources[0] = %+v, want Name=Renamed SeedURL=.../new/", got[0])
	}
	if len(got[0].Tags) != 2 || got[0].Tags[0] != "a" || got[0].Tags[1] != "b" {
		t.Errorf("Tags = %v, want [a b]", got[0].Tags)
	}
}
