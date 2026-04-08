package crawler

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func skipIfShort(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping browser test in short mode (requires Chrome)")
	}
}

func TestBrowser_FetchStaticPage(t *testing.T) {
	skipIfShort(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><body><h1>Static Content</h1><p>Hello from server</p></body></html>`)
	}))
	defer srv.Close()

	b, err := NewBrowser(10 * time.Second)
	if err != nil {
		t.Fatalf("failed to create browser: %v", err)
	}
	defer b.Close()

	body, err := b.Fetch(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("fetch failed: %v", err)
	}

	if !strings.Contains(string(body), "Static Content") {
		t.Errorf("expected 'Static Content' in body, got: %s", string(body))
	}
	if !strings.Contains(string(body), "Hello from server") {
		t.Errorf("expected 'Hello from server' in body, got: %s", string(body))
	}
}

func TestBrowser_FetchJSRenderedPage(t *testing.T) {
	skipIfShort(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><body>
			<div id="content"></div>
			<script>
				document.getElementById('content').innerHTML = '<p>JS Rendered Content</p>';
			</script>
		</body></html>`)
	}))
	defer srv.Close()

	b, err := NewBrowser(10 * time.Second)
	if err != nil {
		t.Fatalf("failed to create browser: %v", err)
	}
	defer b.Close()

	body, err := b.Fetch(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("fetch failed: %v", err)
	}

	if !strings.Contains(string(body), "JS Rendered Content") {
		t.Errorf("expected JS-rendered content in body, got: %s", string(body))
	}
}

func TestBrowser_Timeout(t *testing.T) {
	skipIfShort(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Sleep longer than the test's 1s timeout, but exit early on shutdown
		// so srv.Close() doesn't block for the full duration.
		select {
		case <-time.After(10 * time.Second):
		case <-r.Context().Done():
			return
		}
		fmt.Fprint(w, "slow")
	}))
	defer srv.Close()

	b, err := NewBrowser(1 * time.Second)
	if err != nil {
		t.Fatalf("failed to create browser: %v", err)
	}
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	_, err = b.Fetch(ctx, srv.URL)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestFetch_JSPatternRouting(t *testing.T) {
	htmlServed := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		htmlServed = true
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><body>OK</body></html>`)
	}))
	defer srv.Close()

	// Create fetcher with a JS pattern but no browser — should fall back to static
	f := NewFetcher(5*time.Second, 0, 0)
	f.jsPatterns = []string{"/centers/subject/"}

	result, err := f.Fetch(context.Background(), srv.URL+"/centers/subject/123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !htmlServed {
		t.Error("expected static fetch to be used as fallback")
	}
	if !strings.Contains(string(result.Body), "OK") {
		t.Error("expected body content")
	}
}

func TestFetcher_NeedsJS(t *testing.T) {
	f := NewFetcher(5*time.Second, 0, 0)
	f.jsPatterns = []string{"/centers/subject/", "/dynamic/"}

	tests := []struct {
		url  string
		want bool
	}{
		{"https://www.uma.es/centers/subject/123/456", true},
		{"https://www.uma.es/dynamic/page", true},
		{"https://www.uma.es/etsi-informatica/info/123/", false},
		{"https://www.uma.es/grado-en-ingenieria-informatica/", false},
	}

	for _, tt := range tests {
		got := f.needsJS(tt.url)
		if got != tt.want {
			t.Errorf("needsJS(%q) = %v, want %v", tt.url, got, tt.want)
		}
	}
}
