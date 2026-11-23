package crawler

import (
	"bytes"
	"context"
	"errors"
	"etsi-rag/internal/config"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

const userAgent = "ETSI-RAG-Bot/1.0 (TFG UMA)"

// FetchResult contains the raw response from fetching a URL.
type FetchResult struct {
	URL         string // final URL after redirects (static fetch); the requested URL otherwise
	Body        []byte
	ContentType string // "html" or "pdf"
	StatusCode  int
	Screenshot  []byte // full-page PNG when JS rendering captured one; nil otherwise
}

// Fetcher performs rate-limited HTTP fetching with optional JS rendering.
// Pacing is per lane and shared by every worker that uses the fetcher: a lane
// is a host, or a host's throttled paths, which are paced and held apart.
type Fetcher struct {
	client      *http.Client
	browser     *Browser
	logger      zerolog.Logger
	delay       time.Duration
	maxBodySize int64
	jsPatterns  []string

	throttledPaths     []string
	throttledPathDelay time.Duration

	rateLimitRetries int
	rateLimitBackoff time.Duration
	rateLimitMaxWait time.Duration

	paceMu      sync.Mutex
	nextAllowed map[string]time.Time // lane → earliest start of its next request
}

// NewFetcher creates a fetcher configured from the crawler settings.
func NewFetcher(cfg config.CrawlerConfig, logger zerolog.Logger) *Fetcher {
	f := &Fetcher{client: &http.Client{}, logger: logger, nextAllowed: map[string]time.Time{}}
	f.SetConfig(cfg)
	return f
}

// SetBrowser attaches a headless browser for JS-rendered pages.
func (f *Fetcher) SetBrowser(b *Browser) {
	f.browser = b
}

// SetConfig applies the request, body-size, JS-pattern, pacing and rate-limit settings.
func (f *Fetcher) SetConfig(cfg config.CrawlerConfig) {
	f.client.Timeout = cfg.RequestTimeout
	f.delay = cfg.RequestDelay
	f.maxBodySize = cfg.MaxBodySize
	f.jsPatterns = cfg.JSPatterns
	f.throttledPaths = cfg.ThrottledPaths
	f.throttledPathDelay = cfg.ThrottledPathDelay
	f.rateLimitRetries = cfg.RateLimitRetries
	f.rateLimitBackoff = cfg.RateLimitBackoff
	f.rateLimitMaxWait = cfg.RateLimitMaxWait
}

// Fetch retrieves a URL's contents. Pages matching JS patterns use the
// headless browser if available; otherwise falls back to static HTTP. Requests
// in one lane start at least the lane's delay apart, and a 429 or 503 answer
// holds the whole lane for the backoff wait before the page is retried, up to
// the configured retries.
func (f *Fetcher) Fetch(ctx context.Context, rawURL string) (*FetchResult, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %w", rawURL, err)
	}
	fetch := f.fetchStatic
	if f.needsJS(rawURL) && f.browser != nil {
		fetch = f.fetchDynamic
	}
	lane, delay := f.lane(parsed)
	for attempt := 1; ; attempt++ {
		if err := f.waitTurn(ctx, lane, delay); err != nil {
			return nil, fmt.Errorf("waiting to fetch %s: %w", rawURL, err)
		}
		result, err := fetch(ctx, rawURL)
		var status *statusError
		if !errors.As(err, &status) || !status.rateLimited() {
			return result, err
		}
		wait := f.rateLimitWait(attempt, status.retryAfter)
		f.hold(lane, wait)
		if attempt > f.rateLimitRetries {
			return nil, fmt.Errorf("giving up after %d attempts: %w", attempt, err)
		}
		f.logger.Warn().Str("url", rawURL).Int("status", status.code).Dur("wait", wait).
			Int("attempt", attempt).Msg("rate limited, backing off")
	}
}

// lane is the pacing key and delay for u: the first throttled path pattern its
// path contains, paced at the throttled-path delay, else its host at the
// request delay. A site that limits downloads harder than pages then holds
// only its downloads on a 429, while its pages keep their own pace.
func (f *Fetcher) lane(u *url.URL) (string, time.Duration) {
	path := strings.ToLower(u.Path)
	for _, pattern := range f.throttledPaths {
		if strings.Contains(path, strings.ToLower(pattern)) {
			return u.Host + " " + pattern, f.throttledPathDelay
		}
	}
	return u.Host, f.delay
}

// waitTurn blocks until lane may take its next request, then claims that slot.
// It checks again after each sleep, because a 429 answer to another worker
// may have pushed the lane's next slot back in the meantime.
func (f *Fetcher) waitTurn(ctx context.Context, lane string, delay time.Duration) error {
	for {
		f.paceMu.Lock()
		now := time.Now()
		wait := f.nextAllowed[lane].Sub(now)
		if wait <= 0 {
			f.nextAllowed[lane] = now.Add(delay)
		}
		f.paceMu.Unlock()
		if wait <= 0 {
			return nil
		}
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// hold keeps every worker off lane for at least wait from now.
func (f *Fetcher) hold(lane string, wait time.Duration) {
	f.paceMu.Lock()
	defer f.paceMu.Unlock()
	if until := time.Now().Add(wait); until.After(f.nextAllowed[lane]) {
		f.nextAllowed[lane] = until
	}
}

// statusError is a response whose HTTP status is not a success.
type statusError struct {
	code       int
	retryAfter string // Retry-After header value, empty when absent
}

func (e *statusError) Error() string {
	return fmt.Sprintf("status %d", e.code)
}

// rateLimited reports whether the server asked the client to come back later.
func (e *statusError) rateLimited() bool {
	return e.code == http.StatusTooManyRequests || e.code == http.StatusServiceUnavailable
}

// rateLimitWait is the pause before retry number attempt: the server's
// Retry-After when it parses, else the backoff doubled per earlier retry,
// capped at the max wait either way.
func (f *Fetcher) rateLimitWait(attempt int, retryAfter string) time.Duration {
	if wait, ok := parseRetryAfter(retryAfter); ok {
		return min(wait, f.rateLimitMaxWait)
	}
	wait := f.rateLimitBackoff
	for range attempt - 1 {
		if wait >= f.rateLimitMaxWait {
			break
		}
		wait *= 2
	}
	return min(wait, f.rateLimitMaxWait)
}

// parseRetryAfter reads a Retry-After value in either RFC 9110 form,
// delta-seconds or HTTP-date; a date in the past means no wait.
func parseRetryAfter(v string) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	// 32 bits keeps the seconds-to-Duration conversion from overflowing.
	if secs, err := strconv.ParseUint(v, 10, 32); err == nil {
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(v); err == nil {
		return max(time.Until(t), 0), true
	}
	return 0, false
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
		return nil, fmt.Errorf("fetching %s: %w", rawURL, &statusError{
			code:       resp.StatusCode,
			retryAfter: resp.Header.Get("Retry-After"),
		})
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

	ct, err := contentKind(resp.Header.Get("Content-Type"), body)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", rawURL, err)
	}
	if ct == "html" && isCMSErrorPage(body) {
		return nil, fmt.Errorf("fetching %s: %w", rawURL, errCMSErrorPage)
	}

	return &FetchResult{
		URL:         resp.Request.URL.String(), // after redirects
		Body:        body,
		ContentType: ct,
		StatusCode:  resp.StatusCode,
	}, nil
}

// pdfSniffLen is how far into a body the PDF header may start; PDF readers
// accept it anywhere in the first 1024 bytes.
const pdfSniffLen = 1024

// contentKind classifies a response as "html" or "pdf". The PDF header wins
// over the declared type, because servers often label PDFs
// application/octet-stream; anything else must be declared or sniffed as HTML.
func contentKind(contentType string, body []byte) (string, error) {
	if bytes.Contains(body[:min(len(body), pdfSniffLen)], []byte("%PDF-")) {
		return "pdf", nil
	}
	if contentType == "" {
		contentType = http.DetectContentType(body)
	}
	// The media type comes back even when only a parameter is malformed.
	mediaType, _, _ := mime.ParseMediaType(contentType)
	if mediaType != "text/html" && mediaType != "application/xhtml+xml" {
		return "", fmt.Errorf("unsupported content type %q", contentType)
	}
	return "html", nil
}

// errCMSErrorPage marks UMA's CMS error template, which the site serves with
// status 200, for example for a subject that no longer exists.
var errCMSErrorPage = errors.New("CMS error page")

// isCMSErrorPage reports whether an HTML body is UMA's CMS error template.
func isCMSErrorPage(body []byte) bool {
	return bytes.Contains(body, []byte(`class="httperror-description"`)) ||
		bytes.Contains(body, []byte("<title>Error - Universidad de M"))
}

// fetchDynamic uses the headless browser to render JS, capture the page HTML
// and a full-page PNG screenshot for downstream multimodal retrieval.
func (f *Fetcher) fetchDynamic(ctx context.Context, rawURL string) (*FetchResult, error) {
	body, png, err := f.browser.FetchWithScreenshot(ctx, rawURL)
	if err != nil {
		return nil, fmt.Errorf("dynamic fetch %s: %w", rawURL, err)
	}
	if png == nil {
		f.logger.Warn().Str("url", rawURL).Msg("screenshot failed, page kept without its image")
	}

	if f.maxBodySize > 0 && int64(len(body)) > f.maxBodySize {
		return nil, fmt.Errorf("response from %s exceeds %d bytes", rawURL, f.maxBodySize)
	}
	if isCMSErrorPage(body) {
		return nil, fmt.Errorf("dynamic fetch %s: %w", rawURL, errCMSErrorPage)
	}

	return &FetchResult{
		URL:         rawURL,
		Body:        body,
		ContentType: "html",
		StatusCode:  200,
		Screenshot:  png,
	}, nil
}
