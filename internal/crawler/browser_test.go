package crawler

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
)

func skipIfShort(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping browser test in short mode (requires Chrome)")
	}
}

// newBrowserFetcher returns a fetcher that sends /js/ paths through a
// headless browser bounded by cfg.RequestTimeout.
func newBrowserFetcher(t *testing.T, cfg config.CrawlerConfig) (*Fetcher, *Browser) {
	t.Helper()
	b, err := NewBrowser(cfg.RequestTimeout)
	if err != nil {
		t.Fatalf("failed to create browser: %v", err)
	}
	t.Cleanup(b.Close)
	cfg.JSPatterns = []string{"/js/"}
	f := NewFetcher(cfg, zerolog.Nop())
	f.SetBrowser(b)
	return f, b
}

func TestFetch_BrowserWaitsForLateContentAndScreenshotsFullPage(t *testing.T) {
	skipIfShort(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/js/page", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><body>
			<h1>Static Content</h1>
			<div id="inline"></div>
			<div id="late">Loading...</div>
			<div style="height:3000px"></div>
			<script>
				document.getElementById('inline').innerHTML = '<p>JS Rendered Content</p>';
				window.addEventListener('load', () => setTimeout(() =>
					fetch('/data').then(r => r.text()).then(t => {
						document.getElementById('late').innerHTML = t;
					}), 50));
			</script>
		</body></html>`)
	})
	mux.HandleFunc("/data", func(w http.ResponseWriter, r *http.Request) {
		// Arrives after the load event, as UMA subject pages fill in via AJAX.
		time.Sleep(200 * time.Millisecond)
		for i := range 20 {
			fmt.Fprintf(w, "<p class=\"row%d\">Late row %d</p>", i, i)
		}
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	f, _ := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 30 * time.Second})
	result, err := f.Fetch(context.Background(), srv.URL+"/js/page")
	if err != nil {
		t.Fatalf("fetch failed: %v", err)
	}

	body := string(result.Body)
	for _, want := range []string{"Static Content", "JS Rendered Content", "Late row 0", "Late row 19"} {
		if !strings.Contains(body, want) {
			t.Errorf("expected %q in rendered HTML, got: %s", want, body)
		}
	}

	if result.Screenshot == nil {
		t.Fatal("expected a screenshot")
	}
	img, err := png.Decode(bytes.NewReader(result.Screenshot))
	if err != nil {
		t.Fatalf("screenshot is not a PNG: %v", err)
	}
	if size := img.Bounds().Size(); size.X != emulatedWidth || size.Y < 3000 {
		t.Errorf("expected a full-page %dpx-wide screenshot at least 3000px tall, got %v", emulatedWidth, size)
	}
}

func TestFetch_BrowserStopsOnTimeoutAndCancel(t *testing.T) {
	skipIfShort(t)

	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release) // runs before srv.Close, which waits for handlers

	t.Run("page timeout", func(t *testing.T) {
		f, _ := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 1 * time.Second})
		start := time.Now()
		_, err := f.Fetch(context.Background(), srv.URL+"/js/slow")
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected a deadline error, got %v", err)
		}
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Errorf("fetch took %v, expected it to stop near the 1s timeout", elapsed)
		}
	})

	t.Run("caller cancel", func(t *testing.T) {
		f, _ := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 30 * time.Second})
		ctx, cancel := context.WithCancel(context.Background())
		time.AfterFunc(500*time.Millisecond, cancel)
		start := time.Now()
		_, err := f.Fetch(ctx, srv.URL+"/js/slow")
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected a cancellation error, got %v", err)
		}
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Errorf("fetch took %v, expected it to stop soon after the cancel", elapsed)
		}
	})
}

func TestFetch_BrowserHTTPErrorStatus(t *testing.T) {
	skipIfShort(t)

	cfg := rateLimitConfig(2, 10*time.Millisecond, time.Second)
	cfg.RequestTimeout = 30 * time.Second
	f, _ := newBrowserFetcher(t, cfg)

	tests := []struct {
		name         string
		status       int
		limited      int
		wantErr      []string
		wantRequests int
	}{
		{"404 fails without retry", http.StatusNotFound, -1, []string{"404"}, 1},
		{"429 fails after retries", http.StatusTooManyRequests, -1, []string{"429", "3 attempts"}, 3},
		{"429 then 200 succeeds", http.StatusTooManyRequests, 1, nil, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, requests := newLimitingServer(t, tt.status, "", tt.limited)
			defer srv.Close()

			result, err := f.Fetch(context.Background(), srv.URL+"/js/page")
			if tt.wantErr == nil {
				if err != nil {
					t.Fatalf("expected the retry to succeed, got: %v", err)
				}
				if body := string(result.Body); !strings.Contains(body, "Real content") || strings.Contains(body, "Error page") {
					t.Errorf("expected the page served after the retry, got %s", body)
				}
			} else {
				if err == nil {
					t.Fatalf("expected an error, got a page: %s", result.Body)
				}
				for _, want := range tt.wantErr {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("expected %q in error message, got: %v", want, err)
					}
				}
			}
			if n := len(requests()); n != tt.wantRequests {
				t.Errorf("expected %d requests, got %d", tt.wantRequests, n)
			}
		})
	}
}

func TestFetch_BrowserCMSErrorPageFails(t *testing.T) {
	skipIfShort(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, cmsErrorPage)
	}))
	defer srv.Close()

	f, _ := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 30 * time.Second})
	_, err := f.Fetch(context.Background(), srv.URL+"/js/subject")
	if !errors.Is(err, errCMSErrorPage) {
		t.Fatalf("expected the CMS error page to fail the fetch, got err %v", err)
	}
}

func TestBrowser_ClosesTabsAndProcess(t *testing.T) {
	skipIfShort(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><body><p>Hello</p></body></html>`)
	}))
	defer srv.Close()

	f, b := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 30 * time.Second})
	for i := range 3 {
		if _, err := f.Fetch(context.Background(), fmt.Sprintf("%s/js/%d", srv.URL, i)); err != nil {
			t.Fatalf("fetch %d failed: %v", i, err)
		}
	}

	targets, err := chromedp.Targets(b.ctx)
	if err != nil {
		t.Fatalf("listing targets: %v", err)
	}
	var pages []string
	for _, tg := range targets {
		if tg.Type == "page" {
			pages = append(pages, tg.URL)
		}
	}
	// Only the blank tab Chrome starts with may remain.
	if len(pages) != 1 || pages[0] != "about:blank" {
		t.Errorf("expected only the initial blank tab after the fetches, got %v", pages)
	}

	proc := chromedp.FromContext(b.ctx).Browser.Process()
	b.Close()
	if err := proc.Signal(syscall.Signal(0)); !errors.Is(err, os.ErrProcessDone) {
		t.Errorf("expected the Chrome process to have exited after Close, signal 0 returned %v", err)
	}
}

func TestFetch_BrowserRecoversWhenChromeDies(t *testing.T) {
	skipIfShort(t)

	var requests atomic.Int32
	inFlight := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/js/hang" && requests.Add(1) == 1 {
			// Hold the first request until Chrome is killed under it.
			close(inFlight)
			<-r.Context().Done()
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<html><body><p>Page %s</p></body></html>`, r.URL.Path)
	}))
	defer srv.Close()

	f, b := newBrowserFetcher(t, config.CrawlerConfig{RequestTimeout: 30 * time.Second})
	// killChrome kills the running Chrome and waits until the browser has
	// noticed; it may run off the test goroutine, so it reports with Errorf.
	killChrome := func(t *testing.T) {
		b.mu.Lock()
		ctx := b.ctx
		b.mu.Unlock()
		if err := chromedp.FromContext(ctx).Browser.Process().Kill(); err != nil {
			t.Errorf("killing Chrome: %v", err)
			return
		}
		<-ctx.Done()
	}
	fetchPage := func(t *testing.T, path string) {
		t.Helper()
		result, err := f.Fetch(context.Background(), srv.URL+path)
		if err != nil {
			t.Fatalf("fetch %s: %v", path, err)
		}
		if body := string(result.Body); !strings.Contains(body, "Page "+path) {
			t.Errorf("expected the rendered %s, got %s", path, body)
		}
	}

	t.Run("killed between fetches", func(t *testing.T) {
		fetchPage(t, "/js/before")
		killChrome(t)
		fetchPage(t, "/js/after")
	})

	t.Run("killed during a fetch", func(t *testing.T) {
		killed := make(chan struct{})
		go func() {
			defer close(killed)
			<-inFlight
			killChrome(t)
		}()
		fetchPage(t, "/js/hang")
		<-killed
	})
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
	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
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
	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
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
