package crawler

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const userAgent = "ETSI-RAG-Bot/1.0 (TFG UMA)"

// FetchResult contains the raw response from fetching a URL.
type FetchResult struct {
	URL         string
	Body        []byte
	ContentType string // "html" or "pdf"
	StatusCode  int
	Screenshot  []byte // full-page PNG when JS rendering captured one; nil otherwise
}

// Fetcher performs rate-limited HTTP fetching with optional JS rendering.
type Fetcher struct {
	client      *http.Client
	browser     *Browser
	delay       time.Duration
	maxBodySize int64
	jsPatterns  []string
}

// NewFetcher creates a new rate-limited fetcher.
// maxBodySize of 0 means unlimited.
func NewFetcher(timeout, delay time.Duration, maxBodySize int64) *Fetcher {
	return &Fetcher{
		client:      &http.Client{Timeout: timeout},
		delay:       delay,
		maxBodySize: maxBodySize,
	}
}

// SetBrowser attaches a headless browser for JS-rendered pages.
func (f *Fetcher) SetBrowser(b *Browser) {
	f.browser = b
}

// SetJSPatterns sets URL path prefixes that require JS rendering.
func (f *Fetcher) SetJSPatterns(patterns []string) {
	f.jsPatterns = patterns
}

// SetTimeout updates the HTTP client timeout.
func (f *Fetcher) SetTimeout(d time.Duration) {
	f.client.Timeout = d
}

// SetDelay updates the rate-limiting delay between requests.
func (f *Fetcher) SetDelay(d time.Duration) {
	f.delay = d
}

// SetMaxBody updates the maximum response body size (0 = unlimited).
func (f *Fetcher) SetMaxBody(n int64) {
	f.maxBodySize = n
}

// Fetch retrieves a URL's contents. Pages matching JS patterns use the
// headless browser if available; otherwise falls back to static HTTP.
func (f *Fetcher) Fetch(ctx context.Context, rawURL string) (*FetchResult, error) {
	if f.needsJS(rawURL) && f.browser != nil {
		return f.fetchDynamic(ctx, rawURL)
	}
	return f.fetchStatic(ctx, rawURL)
}

// needsJS returns true if the URL path matches any configured JS pattern.
func (f *Fetcher) needsJS(rawURL string) bool {
	if len(f.jsPatterns) == 0 {
		return false
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	path := strings.ToLower(parsed.Path)
	for _, pattern := range f.jsPatterns {
		if strings.Contains(path, strings.ToLower(pattern)) {
			return true
		}
	}
	return false
}

// fetchStatic performs a plain HTTP GET.
func (f *Fetcher) fetchStatic(ctx context.Context, rawURL string) (*FetchResult, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", rawURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: status %d", rawURL, resp.StatusCode)
	}

	var reader io.Reader = resp.Body
	if f.maxBodySize > 0 {
		reader = io.LimitReader(resp.Body, f.maxBodySize+1)
	}

	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("reading response from %s: %w", rawURL, err)
	}

	if f.maxBodySize > 0 && int64(len(body)) > f.maxBodySize {
		return nil, fmt.Errorf("response from %s exceeds %d bytes", rawURL, f.maxBodySize)
	}

	ct := "html"
	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(contentType, "pdf") || strings.HasSuffix(strings.ToLower(rawURL), ".pdf") {
		ct = "pdf"
	}

	return &FetchResult{
		URL:         rawURL,
		Body:        body,
		ContentType: ct,
		StatusCode:  resp.StatusCode,
	}, nil
}

// fetchDynamic uses the headless browser to render JS, capture the page HTML
// and a full-page PNG screenshot for downstream multimodal retrieval.
func (f *Fetcher) fetchDynamic(ctx context.Context, rawURL string) (*FetchResult, error) {
	body, png, err := f.browser.FetchWithScreenshot(ctx, rawURL)
	if err != nil {
		return nil, fmt.Errorf("dynamic fetch %s: %w", rawURL, err)
	}

	if f.maxBodySize > 0 && int64(len(body)) > f.maxBodySize {
		return nil, fmt.Errorf("response from %s exceeds %d bytes", rawURL, f.maxBodySize)
	}

	return &FetchResult{
		URL:         rawURL,
		Body:        body,
		ContentType: "html",
		StatusCode:  200,
		Screenshot:  png,
	}, nil
}

// Delay returns the configured delay between requests.
func (f *Fetcher) Delay() time.Duration {
	return f.delay
}
