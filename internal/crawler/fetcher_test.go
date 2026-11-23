package crawler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
)

func TestFetch_HTMLSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != userAgent {
			t.Errorf("expected User-Agent %q, got %q", userAgent, r.Header.Get("User-Agent"))
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte("<html><body>Hello</body></html>"))
	}))
	defer srv.Close()

	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
	result, err := f.Fetch(context.Background(), srv.URL+"/page")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "html" {
		t.Errorf("expected content type html, got %s", result.ContentType)
	}
	if result.StatusCode != 200 {
		t.Errorf("expected status 200, got %d", result.StatusCode)
	}
	if !strings.Contains(string(result.Body), "Hello") {
		t.Errorf("expected body to contain 'Hello', got %s", string(result.Body))
	}
}

func TestFetch_PDFDetection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		w.Write([]byte("%PDF-1.4 fake content"))
	}))
	defer srv.Close()

	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
	result, err := f.Fetch(context.Background(), srv.URL+"/doc.pdf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "pdf" {
		t.Errorf("expected content type pdf, got %s", result.ContentType)
	}
}

func TestFetch_PDFSuffixDoesNotDecideKind(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<!DOCTYPE html><html><body>Documento no disponible</body></html>"))
	}))
	defer srv.Close()

	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
	result, err := f.Fetch(context.Background(), srv.URL+"/document.pdf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "html" {
		t.Errorf("an HTML body at a .pdf URL got content type %q, want html", result.ContentType)
	}
}

func TestFetch_ContentKind(t *testing.T) {
	tests := []struct {
		name        string
		contentType string // "" lets the server sniff it
		body        string
		want        string // "" means the fetch must fail
	}{
		{"html with charset", "text/html; charset=utf-8", "<html><body>Hi</body></html>", "html"},
		{"xhtml", "application/xhtml+xml", "<html><body>Hi</body></html>", "html"},
		{"undeclared html", "", "<!DOCTYPE html><html><body>Hi</body></html>", "html"},
		{"pdf", "application/pdf", "%PDF-1.7 content", "pdf"},
		{"pdf labelled octet-stream", "application/octet-stream", "%PDF-1.4 content", "pdf"},
		{"pdf header after leading bytes", "application/octet-stream", "\r\n%PDF-1.4 content", "pdf"},
		{"certificate", "application/pkix-cert", "0\x82\x05\x8a0\x82\x04r", ""},
		{"configuration profile", "application/x-apple-aspen-config", `<?xml version="1.0"?><plist></plist>`, ""},
		{"plain text", "text/plain", "just text", ""},
		{"html labelled pdf", "application/pdf", "<html><body>Not found</body></html>", ""},
		{"html with a malformed parameter", "text/html; charset", "<html></html>", "html"},
		{"malformed type", "text/", "<html></html>", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.contentType != "" {
					w.Header().Set("Content-Type", tt.contentType)
				}
				io.WriteString(w, tt.body)
			}))
			defer srv.Close()

			f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
			// No .pdf suffix: the kind must come from the response alone.
			result, err := f.Fetch(context.Background(), srv.URL+"/file")
			if tt.want == "" {
				if err == nil {
					t.Fatalf("expected the fetch to fail, got content type %q", result.ContentType)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.ContentType != tt.want {
				t.Errorf("content type = %q, want %q", result.ContentType, tt.want)
			}
		})
	}
}

func TestFetch_BodySizeLimit(t *testing.T) {
	largeBody := strings.Repeat("x", 1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(largeBody))
	}))
	defer srv.Close()

	// Limit to 512 bytes
	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second, MaxBodySize: 512}, zerolog.Nop())
	_, err := f.Fetch(context.Background(), srv.URL+"/big")
	if err == nil {
		t.Fatal("expected error for body exceeding size limit")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected 'exceeds' in error message, got: %v", err)
	}
}

func TestFetch_BodySizeLimitZeroUnlimited(t *testing.T) {
	body := strings.Repeat("x", 4096)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(body))
	}))
	defer srv.Close()

	// 0 = unlimited
	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
	result, err := f.Fetch(context.Background(), srv.URL+"/big")
	if err != nil {
		t.Fatalf("unexpected error with unlimited body size: %v", err)
	}
	if len(result.Body) != 4096 {
		t.Errorf("expected 4096 bytes, got %d", len(result.Body))
	}
}

func TestFetch_StatusErrorIsNotRetried(t *testing.T) {
	srv, requests := newLimitingServer(t, http.StatusNotFound, "", -1)
	defer srv.Close()

	f := NewFetcher(rateLimitConfig(4, time.Millisecond, time.Second), zerolog.Nop())
	_, err := f.Fetch(context.Background(), srv.URL+"/missing")
	if err == nil {
		t.Fatal("expected error for 404 status")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("expected '404' in error message, got: %v", err)
	}
	if n := len(requests()); n != 1 {
		t.Errorf("expected a 404 to be fetched once, got %d requests", n)
	}
}

// newLimitingServer answers the first `limited` requests (every request when
// negative) with status and an optional Retry-After header, then serves an
// HTML page. requests returns the arrival time of each request so far.
func newLimitingServer(t *testing.T, status int, retryAfter string, limited int) (srv *httptest.Server, requests func() []time.Time) {
	t.Helper()
	var mu sync.Mutex
	var times []time.Time
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/favicon.ico" { // requested by the browser, not the crawler
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		times = append(times, time.Now())
		n := len(times)
		mu.Unlock()
		code, body := http.StatusOK, "<html><body>Real content</body></html>"
		if limited < 0 || n <= limited {
			if retryAfter != "" {
				w.Header().Set("Retry-After", retryAfter)
			}
			code, body = status, fmt.Sprintf("<html><body>Error page %d</body></html>", status)
		}
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(code)
		if _, err := io.WriteString(w, body); err != nil {
			t.Errorf("writing response: %v", err)
		}
	}))
	return srv, func() []time.Time {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(times)
	}
}

func rateLimitConfig(retries int, backoff, maxWait time.Duration) config.CrawlerConfig {
	return config.CrawlerConfig{
		RequestTimeout:   5 * time.Second,
		RateLimitRetries: retries,
		RateLimitBackoff: backoff,
		RateLimitMaxWait: maxWait,
	}
}

func TestFetch_RetriesRateLimitedPageAfterRetryAfter(t *testing.T) {
	past := time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat)
	tests := []struct {
		name       string
		status     int
		retryAfter string
	}{
		{"429 delta-seconds", http.StatusTooManyRequests, "0"},
		{"503 delta-seconds", http.StatusServiceUnavailable, "0"},
		{"429 HTTP-date", http.StatusTooManyRequests, past},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, requests := newLimitingServer(t, tt.status, tt.retryAfter, 1)
			defer srv.Close()

			// The hour-long backoff only finishes inside the deadline if the
			// Retry-After value replaces it.
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			f := NewFetcher(rateLimitConfig(4, time.Hour, time.Hour), zerolog.Nop())
			result, err := f.Fetch(ctx, srv.URL+"/page")
			if err != nil {
				t.Fatalf("expected the retry to succeed, got: %v", err)
			}
			if !strings.Contains(string(result.Body), "Real content") {
				t.Errorf("expected the page served after the retry, got %s", result.Body)
			}
			if n := len(requests()); n != 2 {
				t.Errorf("expected 2 requests, got %d", n)
			}
		})
	}
}

func TestFetch_RateLimitWaitCappedAtMaxWait(t *testing.T) {
	const maxWait = 100 * time.Millisecond
	future := time.Now().Add(time.Hour).UTC().Format(http.TimeFormat)
	tests := []struct {
		name       string
		retryAfter string
		backoff    time.Duration
	}{
		// A tiny backoff shows the wait came from Retry-After.
		{"Retry-After delta-seconds", "3600", time.Millisecond},
		{"Retry-After HTTP-date", future, time.Millisecond},
		{"backoff", "", time.Hour},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, requests := newLimitingServer(t, http.StatusTooManyRequests, tt.retryAfter, 1)
			defer srv.Close()

			f := NewFetcher(rateLimitConfig(1, tt.backoff, maxWait), zerolog.Nop())
			if _, err := f.Fetch(context.Background(), srv.URL+"/page"); err != nil {
				t.Fatalf("expected the retry to succeed, got: %v", err)
			}
			times := requests()
			if len(times) != 2 {
				t.Fatalf("expected 2 requests, got %d", len(times))
			}
			if gap := times[1].Sub(times[0]); gap < maxWait || gap > 3*time.Second {
				t.Errorf("expected the retry about %v after the 429, got %v", maxWait, gap)
			}
		})
	}
}

func TestFetch_RateLimitRetriesBackOffExponentiallyThenGiveUp(t *testing.T) {
	const backoff = 20 * time.Millisecond
	srv, requests := newLimitingServer(t, http.StatusTooManyRequests, "", -1)
	defer srv.Close()

	f := NewFetcher(rateLimitConfig(3, backoff, time.Second), zerolog.Nop())
	_, err := f.Fetch(context.Background(), srv.URL+"/page")
	if err == nil {
		t.Fatal("expected an error once the retries run out")
	}
	for _, want := range []string{"429", "4 attempts"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("expected %q in error message, got: %v", want, err)
		}
	}
	times := requests()
	if len(times) != 4 {
		t.Fatalf("expected 1 request and 3 retries, got %d requests", len(times))
	}
	for i := 1; i < len(times); i++ {
		want := backoff << (i - 1)
		if gap := times[i].Sub(times[i-1]); gap < want {
			t.Errorf("retry %d came %v after the previous request, expected at least %v", i, gap, want)
		}
	}
}

func TestFetch_RateLimitWaitStopsOnCancel(t *testing.T) {
	srv, requests := newLimitingServer(t, http.StatusTooManyRequests, "", -1)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(100*time.Millisecond, cancel)
	f := NewFetcher(rateLimitConfig(4, time.Hour, time.Hour), zerolog.Nop())
	start := time.Now()
	_, err := f.Fetch(ctx, srv.URL+"/page")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected a cancellation error, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("fetch took %v, expected it to stop soon after the cancel", elapsed)
	}
	if n := len(requests()); n != 1 {
		t.Errorf("expected no retry after the cancel, got %d requests", n)
	}
}

func TestFetch_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.Write([]byte("slow"))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second}, zerolog.Nop())
	_, err := f.Fetch(ctx, srv.URL+"/slow")
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestFetch_BodySizeLimitExact(t *testing.T) {
	// Body exactly at limit should succeed
	body := strings.Repeat("x", 512)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(body))
	}))
	defer srv.Close()

	f := NewFetcher(config.CrawlerConfig{RequestTimeout: 5 * time.Second, MaxBodySize: 512}, zerolog.Nop())
	result, err := f.Fetch(context.Background(), srv.URL+"/exact")
	if err != nil {
		t.Fatalf("body at exact limit should succeed: %v", err)
	}
	if len(result.Body) != 512 {
		t.Errorf("expected 512 bytes, got %d", len(result.Body))
	}
}

// TestFetch_ThrottledPathsHaveTheirOwnLane checks that downloads under a
// throttled path are paced at the throttled delay, and that a 429 on one holds
// only the downloads: pages on the same host keep their own pace meanwhile.
func TestFetch_ThrottledPathsHaveTheirOwnLane(t *testing.T) {
	const (
		throttledDelay = 300 * time.Millisecond
		backoff        = 2 * time.Second
	)
	var mu sync.Mutex
	var downloads []time.Time
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		if strings.Contains(r.URL.Path, "/navegador_de_ficheros/") {
			mu.Lock()
			downloads = append(downloads, time.Now())
			first := len(downloads) == 1
			mu.Unlock()
			if first {
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
		}
		io.WriteString(w, "<html><body>Real content</body></html>")
	}))
	defer srv.Close()

	cfg := rateLimitConfig(2, backoff, backoff)
	cfg.RequestDelay = time.Millisecond
	cfg.ThrottledPaths = []string{"/Navegador_de_Ficheros/"}
	cfg.ThrottledPathDelay = throttledDelay
	f := NewFetcher(cfg, zerolog.Nop())
	ctx := context.Background()

	start := time.Now()
	var wg sync.WaitGroup
	wg.Go(func() {
		if _, err := f.Fetch(ctx, srv.URL+"/etsi/navegador_de_ficheros/descargar/a.pdf"); err != nil {
			t.Errorf("throttled download a: %v", err)
		}
	})
	time.Sleep(100 * time.Millisecond) // the first download is answered 429 by now
	for i := range 3 {
		if _, err := f.Fetch(ctx, fmt.Sprintf("%s/etsi/page%d/", srv.URL, i)); err != nil {
			t.Fatalf("page %d: %v", i, err)
		}
	}
	if elapsed := time.Since(start); elapsed >= backoff {
		t.Errorf("pages finished %v after the download's 429, want them not held by its %v backoff", elapsed, backoff)
	}
	wg.Wait()
	if _, err := f.Fetch(ctx, srv.URL+"/etsi/navegador_de_ficheros/descargar/b.pdf"); err != nil {
		t.Fatalf("throttled download b: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(downloads) != 3 {
		t.Fatalf("expected 3 download requests (a, a's retry, b), got %d", len(downloads))
	}
	if gap := downloads[1].Sub(downloads[0]); gap < backoff-50*time.Millisecond {
		t.Errorf("download retried %v after its 429, want at least the %v backoff", gap, backoff)
	}
	if gap := downloads[2].Sub(downloads[1]); gap < throttledDelay-50*time.Millisecond {
		t.Errorf("downloads %v apart, want at least the throttled delay %v", gap, throttledDelay)
	}
}
