package crawler

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
)

// Browser provides headless Chrome page fetching for JS-rendered pages.
type Browser struct {
	browser *rod.Browser
	timeout time.Duration
}

// NewBrowser creates a headless Chrome browser instance.
// The browser binary is auto-detected or downloaded.
func NewBrowser(timeout time.Duration) (*Browser, error) {
	path, _ := launcher.LookPath()
	u := launcher.New().Bin(path).Headless(true).MustLaunch()

	browser := rod.New().ControlURL(u)
	if err := browser.Connect(); err != nil {
		return nil, fmt.Errorf("connecting to browser: %w", err)
	}

	return &Browser{
		browser: browser,
		timeout: timeout,
	}, nil
}

// Fetch navigates to a URL, waits for the page to settle, and returns the rendered HTML.
func (b *Browser) Fetch(ctx context.Context, pageURL string) ([]byte, error) {
	page, err := b.browser.Page(proto.TargetCreateTarget{URL: ""})
	if err != nil {
		return nil, fmt.Errorf("creating page: %w", err)
	}
	defer page.Close()

	page = page.Context(ctx).Timeout(b.timeout)

	if err := page.Navigate(pageURL); err != nil {
		return nil, fmt.Errorf("navigating to %s: %w", pageURL, err)
	}

	if err := page.WaitDOMStable(500*time.Millisecond, 0.1); err != nil {
		return nil, fmt.Errorf("waiting for DOM stable on %s: %w", pageURL, err)
	}

	html, err := page.HTML()
	if err != nil {
		return nil, fmt.Errorf("getting HTML from %s: %w", pageURL, err)
	}

	return []byte(html), nil
}

// Close shuts down the browser.
func (b *Browser) Close() {
	if b.browser != nil {
		b.browser.Close()
	}
}
