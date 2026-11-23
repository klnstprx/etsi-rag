package crawler

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
)

// seedSources is one enabled source per seed URL.
func seedSources(seeds ...string) []domain.CrawlSource {
	var srcs []domain.CrawlSource
	for i, seed := range seeds {
		key := fmt.Sprintf("src%d", i)
		srcs = append(srcs, domain.CrawlSource{Key: key, Name: key, SeedURL: seed, Enabled: true})
	}
	return srcs
}

// runConfig is a crawler config for small test sites with fast rate-limit
// retries; tests override the fields they exercise.
func runConfig() config.CrawlerConfig {
	return config.CrawlerConfig{
		RequestDelay:        time.Millisecond,
		RequestTimeout:      5 * time.Second,
		MaxDepth:            3,
		MaxPages:            100,
		Concurrency:         1,
		ParseWorkers:        1,
		EmbedWorkers:        1,
		MaxEmbedFails:       2,
		ProgressLogInterval: time.Hour,
		RateLimitRetries:    1,
		RateLimitBackoff:    time.Millisecond,
		RateLimitMaxWait:    time.Millisecond,
	}
}

func pageStatuses(t *testing.T, s *store.SQLiteStore) map[string]domain.CrawlPage {
	t.Helper()
	pages, err := s.ListCrawlPages(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("ListCrawlPages: %v", err)
	}
	byURL := make(map[string]domain.CrawlPage, len(pages))
	for _, p := range pages {
		byURL[p.URL] = p
	}
	return byURL
}

func TestRunAll_PacesRequestsPerHostAcrossSources(t *testing.T) {
	const (
		delay   = 100 * time.Millisecond
		backoff = 500 * time.Millisecond
		// Arrival times are taken by the server, so allow for scheduling jitter.
		slack = 20 * time.Millisecond
	)
	var mu sync.Mutex
	var arrivals []time.Time
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		arrivals = append(arrivals, time.Now())
		n := len(arrivals)
		mu.Unlock()
		if n == 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<html><body><main><p>Page %s</p></main></body></html>`, r.URL.Path)
	}))
	defer srv.Close()

	var seeds []string
	for i := range 4 {
		seeds = append(seeds, fmt.Sprintf("%s/s%d/", srv.URL, i))
	}
	cfg := runConfig()
	cfg.Concurrency = len(seeds)
	cfg.RequestDelay = delay
	cfg.RateLimitBackoff = backoff
	cfg.RateLimitMaxWait = backoff
	c, s, _ := newRunHarness(t, cfg, stagesTestEmbedder{}, seedSources(seeds...))

	if _, err := c.RunAll(context.Background()); err != nil {
		t.Fatalf("RunAll: %v", err)
	}

	mu.Lock()
	times := arrivals
	mu.Unlock()
	if len(times) != len(seeds)+1 {
		t.Fatalf("expected %d requests (one per seed plus the retry), got %d", len(seeds)+1, len(times))
	}
	for i := 1; i < len(times); i++ {
		if gap := times[i].Sub(times[i-1]); gap < delay-slack {
			t.Errorf("requests %d and %d to one host came %v apart, want at least %v", i, i+1, gap, delay)
		}
	}
	limitedAt := times[1]
	for i, at := range times[2:] {
		if gap := at.Sub(limitedAt); gap < backoff-slack {
			t.Errorf("request %d came %v after the 429, want no request to the host for %v", i+3, gap, backoff)
		}
	}
	pages := pageStatuses(t, s)
	for _, seed := range seeds {
		if p := pages[seed]; p.Status != domain.CrawlPageFetched {
			t.Errorf("page %s: status %q, want %q", seed, p.Status, domain.CrawlPageFetched)
		}
	}
}

// cmsErrorPage is the shape of the error template UMA's CMS serves with
// status 200, for example for /centers/subject/ pages of retired subjects.
const cmsErrorPage = `<!DOCTYPE html><html lang="es"><head>
<title>Error - Universidad de Málaga - Universidad de Málaga</title></head>
<body><div class="row"><div class="col-xs-12">
<h1>Lo sentimos, no podemos cargar esta p&aacute;gina</h1>
<p class="httperror-description">
<p>Ha ocurrido un error al intentar obtener los datos y no ha sido posible cargar esta p&aacute;gina.</p>
</p></div></div></body></html>`

func TestRunAll_CMSErrorPageIsAFailedFetch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, cmsErrorPage)
	}))
	defer srv.Close()

	seed := srv.URL + "/centers/subject/5388/1/"
	c, s, _ := newRunHarness(t, runConfig(), stagesTestEmbedder{}, seedSources(seed))
	if _, err := c.RunAll(context.Background()); err != nil {
		t.Fatalf("RunAll: %v", err)
	}

	p, ok := pageStatuses(t, s)[seed]
	if !ok {
		t.Fatalf("expected a crawl_pages row for %s", seed)
	}
	if p.Status != domain.CrawlPageFailed {
		t.Errorf("page status %q, want %q", p.Status, domain.CrawlPageFailed)
	}
	if p.DocID != nil || p.ErrorMessage == nil || !strings.Contains(*p.ErrorMessage, "CMS error page") {
		t.Errorf("expected a page with no document and an error naming the CMS error page, got doc %v, error %v", p.DocID, p.ErrorMessage)
	}
	docs, err := s.ListDocuments(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListDocuments: %v", err)
	}
	if len(docs) != 0 {
		t.Errorf("expected no document from the error page, got %d (first titled %q)", len(docs), docs[0].Title)
	}
}
