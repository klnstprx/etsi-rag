package admin

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
)

// setupTestHandlerWithSources extends setupTestAdminHandler with a writable
// configDir seeded with one source, so the source CRUD handlers have somewhere
// to read and write.
func setupTestHandlerWithSources(t *testing.T, seed []domain.CrawlSource) *Handler {
	t.Helper()
	h, _ := setupTestAdminHandler(t)

	dir := t.TempDir()
	h.configDir = dir
	if err := config.WriteSources(seed, config.SourcesPath(dir)); err != nil {
		t.Fatalf("seeding sources.toml: %v", err)
	}
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

	got, err := config.LoadSources(h.configDir)
	if err != nil {
		t.Fatalf("LoadSources: %v", err)
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

// stat helper exists so a future failing happy-path test can also assert the
// file exists at the expected location, surfacing tempdir wiring breakage
// before the test author has to chase it.
func TestSourceUpdate_HappyPath_FileLayout(t *testing.T) {
	h := setupTestHandlerWithSources(t, []domain.CrawlSource{
		{Key: "valid", Name: "X", SeedURL: "https://example.com/", Enabled: true},
	})
	expected := filepath.Join(h.configDir, "sources.toml")
	if _, err := os.Stat(expected); err != nil {
		t.Fatalf("expected sources.toml at %s: %v", expected, err)
	}
	if !strings.HasPrefix(h.configDir, os.TempDir()) {
		// Defensive: future setup change might point configDir at $HOME.
		t.Logf("configDir = %s (not under TempDir; verify before extending)", h.configDir)
	}
}
