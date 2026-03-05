package crawler

import (
	"context"
	"fmt"
	"io"
	"net/http"
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
}

// Fetcher performs rate-limited HTTP fetching.
type Fetcher struct {
	client *http.Client
	delay  time.Duration
}

// NewFetcher creates a new rate-limited fetcher.
func NewFetcher(timeout, delay time.Duration) *Fetcher {
	return &Fetcher{
		client: &http.Client{Timeout: timeout},
		delay:  delay,
	}
}

// Fetch retrieves a URL's contents. The caller must manage rate-limiting
// by calling this in a ticker-gated loop.
func (f *Fetcher) Fetch(ctx context.Context, url string) (*FetchResult, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: status %d", url, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response from %s: %w", url, err)
	}

	ct := "html"
	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(contentType, "pdf") || strings.HasSuffix(strings.ToLower(url), ".pdf") {
		ct = "pdf"
	}

	return &FetchResult{
		URL:         url,
		Body:        body,
		ContentType: ct,
		StatusCode:  resp.StatusCode,
	}, nil
}

// Delay returns the configured delay between requests.
func (f *Fetcher) Delay() time.Duration {
	return f.delay
}
