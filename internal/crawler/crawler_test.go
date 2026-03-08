package crawler

import (
	"strings"
	"testing"
)

func TestTitleFromURL(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		{"pdf filename", "https://sara.uma.es/ht/2025/ProgramasAsignaturas_5102.pdf", "ProgramasAsignaturas_5102"},
		{"html path segment", "https://www.uma.es/etsi-informatica/info/plan-estudios", "plan-estudios"},
		{"trailing slash", "https://www.uma.es/etsi-informatica/info/", "info"},
		{"root path only", "https://www.uma.es/", "https://www.uma.es/"},
		{"no path", "https://www.uma.es", "https://www.uma.es"},
		{"query string ignored", "https://example.com/doc.pdf?v=2", "doc"},
		{"deeply nested", "https://example.com/a/b/c/d/file.pdf", "file"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := titleFromURL(tt.url)
			if got != tt.want {
				t.Errorf("titleFromURL(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}

func TestObjectStoreKey(t *testing.T) {
	t.Run("html prefix", func(t *testing.T) {
		key := objectStoreKey("html", "https://www.uma.es/page")
		if !strings.HasPrefix(key, "crawl/html/") {
			t.Errorf("expected crawl/html/ prefix, got %q", key)
		}
	})

	t.Run("pdf prefix", func(t *testing.T) {
		key := objectStoreKey("pdf", "https://example.com/doc.pdf")
		if !strings.HasPrefix(key, "crawl/pdf/") {
			t.Errorf("expected crawl/pdf/ prefix, got %q", key)
		}
	})

	t.Run("deterministic", func(t *testing.T) {
		k1 := objectStoreKey("html", "https://www.uma.es/page")
		k2 := objectStoreKey("html", "https://www.uma.es/page")
		if k1 != k2 {
			t.Errorf("not deterministic: %q != %q", k1, k2)
		}
	})

	t.Run("different URLs produce different keys", func(t *testing.T) {
		k1 := objectStoreKey("html", "https://www.uma.es/page1")
		k2 := objectStoreKey("html", "https://www.uma.es/page2")
		if k1 == k2 {
			t.Errorf("different URLs produced same key: %q", k1)
		}
	})
}
