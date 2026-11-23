package crawler

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/cdp"
	"github.com/chromedp/cdproto/domsnapshot"
	"github.com/chromedp/cdproto/emulation"
	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/chromedp"
)

// A page has settled once two DOM snapshots taken domStableInterval apart
// differ by at most domStableMaxDiff (the go-rod WaitDOMStable rule).
const (
	domStableInterval = 500 * time.Millisecond
	domStableMaxDiff  = 0.1
)

// The device go-rod emulated by default (LaptopWithMDPIScreen, landscape),
// kept so pages are requested and rendered as when the corpus was crawled.
const (
	emulatedWidth          = 1280
	emulatedHeight         = 800
	emulatedUserAgent      = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
	emulatedAcceptLanguage = "en"
)

// Browser provides headless Chrome page fetching for JS-rendered pages.
type Browser struct {
	timeout time.Duration

	mu     sync.Mutex
	ctx    context.Context // chromedp context that owns the Chrome process; cancelled when Chrome dies
	cancel context.CancelFunc
	closed bool
}

// NewBrowser starts a headless Chrome found on the system (Chrome or
// Chromium in the standard locations or on $PATH).
func NewBrowser(timeout time.Duration) (*Browser, error) {
	ctx, cancel, err := launchChrome()
	if err != nil {
		return nil, err
	}
	return &Browser{ctx: ctx, cancel: cancel, timeout: timeout}, nil
}

// launchChrome starts Chrome and returns the context that owns its process;
// cancel stops Chrome and waits for it to exit.
func launchChrome() (context.Context, context.CancelFunc, error) {
	var opts []chromedp.ContextOption
	if inContainer() {
		// Container runtimes commonly block the namespaces Chrome's sandbox
		// needs; go-rod's launcher passed --no-sandbox in the same case.
		opts = append(opts, chromedp.WithAllocatorOptions(chromedp.NoSandbox))
	}
	ctx, cancel := chromedp.NewContext(context.Background(), opts...)
	// The first run starts Chrome and ties its lifetime to ctx, so it must
	// not carry a timeout.
	if err := chromedp.Do(ctx); err != nil {
		cancel()
		return nil, nil, fmt.Errorf("launching browser: %w", err)
	}
	return ctx, cancel, nil
}

// live returns the context of a running Chrome, relaunching Chrome first if
// it has died since the last fetch.
func (b *Browser) live() (context.Context, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil, errors.New("browser closed")
	}
	if b.ctx.Err() == nil {
		return b.ctx, nil
	}
	b.cancel()
	ctx, cancel, err := launchChrome()
	if err != nil {
		return nil, fmt.Errorf("relaunching after Chrome exited: %w", err)
	}
	b.ctx, b.cancel = ctx, cancel
	return ctx, nil
}

// FetchWithScreenshot navigates to a URL in a new tab, waits for the page to
// settle, and returns the rendered HTML and a full-page PNG screenshot. A
// failed screenshot is not an error: the PNG is nil and the HTML is returned.
// If Chrome dies during the fetch, it is relaunched and the page tried once more.
func (b *Browser) FetchWithScreenshot(ctx context.Context, pageURL string) (html []byte, png []byte, err error) {
	browserCtx, err := b.live()
	if err != nil {
		return nil, nil, err
	}
	html, png, err = b.fetch(ctx, browserCtx, pageURL)
	if err != nil && browserCtx.Err() != nil && ctx.Err() == nil {
		if browserCtx, err = b.live(); err != nil {
			return nil, nil, err
		}
		return b.fetch(ctx, browserCtx, pageURL)
	}
	return html, png, err
}

func (b *Browser) fetch(ctx, browserCtx context.Context, pageURL string) (html []byte, png []byte, err error) {
	tab, closeTab := chromedp.NewContext(browserCtx)
	defer closeTab()
	// A tab must descend from the browser context, so the caller's ctx
	// cancels it through AfterFunc instead of parenthood.
	tab, cancel := context.WithTimeout(tab, b.timeout)
	defer cancel()
	defer context.AfterFunc(ctx, cancel)()

	if err := chromedp.Do(tab, emulateDevice()); err != nil {
		return nil, nil, fmt.Errorf("emulating device for %s: %w", pageURL, err)
	}
	resp, err := chromedp.Run(tab, chromedp.NavigateResponse(pageURL))
	if err != nil {
		return nil, nil, fmt.Errorf("navigating to %s: %w", pageURL, err)
	}
	if resp == nil {
		return nil, nil, fmt.Errorf("navigating to %s: no response for the document", pageURL)
	}
	// An error page renders like any other, so only the status tells it apart.
	if resp.Status < 200 || resp.Status > 299 {
		return nil, nil, fmt.Errorf("navigating to %s: %w", pageURL, &statusError{
			code:       int(resp.Status),
			retryAfter: headerValue(resp.Headers, "Retry-After"),
		})
	}
	if err := chromedp.Do(tab, waitDOMStable(domStableInterval, domStableMaxDiff)); err != nil {
		return nil, nil, fmt.Errorf("waiting for DOM stable on %s: %w", pageURL, err)
	}
	doc, err := chromedp.Run(tab, chromedp.OuterHTML(chromedp.CSS("html")))
	if err != nil {
		return nil, nil, fmt.Errorf("getting HTML from %s: %w", pageURL, err)
	}

	shot, err := chromedp.Run(tab, chromedp.FullScreenshot(100)) // quality 100 selects PNG
	if err != nil {
		// Non-fatal: the page falls back to text-only retrieval.
		return []byte(doc), nil, nil
	}
	return []byte(doc), shot, nil
}

// Close shuts down the browser and waits for the Chrome process to exit.
func (b *Browser) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	b.cancel()
}

// headerValue looks up a CDP response header, whose key keeps the server's
// casing.
func headerValue(h network.Headers, name string) string {
	for k, v := range h {
		if strings.EqualFold(k, name) {
			s, _ := v.(string)
			return s
		}
	}
	return ""
}

// inContainer reports whether the process runs in a Docker, Podman or
// Kubernetes container, detected as go-rod did.
func inContainer() bool {
	for _, marker := range []string{"/.dockerenv", "/.containerenv"} {
		if _, err := os.Stat(marker); err == nil {
			return true
		}
	}
	return os.Getenv("KUBERNETES_SERVICE_HOST") != ""
}

func emulateDevice() chromedp.Action[chromedp.Void] {
	return chromedp.Steps(
		chromedp.EmulateViewport(emulatedWidth, emulatedHeight, chromedp.EmulateLandscape),
		chromedp.Func(func(ctx context.Context, t *chromedp.Target) error {
			_, err := cdp.Call(ctx, t, emulation.SetUserAgentOverride, emulation.SetUserAgentOverrideParams{
				UserAgent:      emulatedUserAgent,
				AcceptLanguage: emulatedAcceptLanguage,
			})
			return err
		}),
	)
}

// waitDOMStable polls the DOM snapshot string table every interval until two
// consecutive snapshots differ by at most maxDiff (see domDiff).
func waitDOMStable(interval time.Duration, maxDiff float64) chromedp.Action[chromedp.Void] {
	return chromedp.Func(func(ctx context.Context, t *chromedp.Target) error {
		prev, err := domStrings(ctx, t)
		if err != nil {
			return err
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
			case <-ctx.Done():
				return ctx.Err()
			}
			cur, err := domStrings(ctx, t)
			if err != nil {
				return err
			}
			if domDiff(prev, cur) <= maxDiff {
				return nil
			}
			prev = cur
		}
	})
}

// domStrings captures the DOM snapshot with the same options go-rod used and
// returns its shared string table (tag names, text, attributes, colours).
func domStrings(ctx context.Context, t *chromedp.Target) ([]string, error) {
	yes := true
	res, err := cdp.Call(ctx, t, domsnapshot.CaptureSnapshot, domsnapshot.CaptureSnapshotParams{
		ComputedStyles:                 []string{},
		IncludePaintOrder:              &yes,
		IncludeDOMRects:                &yes,
		IncludeBlendedBackgroundColors: &yes,
		IncludeTextColorOpacities:      &yes,
	})
	if err != nil {
		return nil, fmt.Errorf("capturing DOM snapshot: %w", err)
	}
	return res.Strings, nil
}

// domDiff returns the fraction of cur that is not part of the longest common
// subsequence of prev and cur: 0 when equal, 1 when nothing is shared.
func domDiff(prev, cur []string) float64 {
	if len(cur) == 0 {
		if len(prev) == 0 {
			return 0
		}
		return 1
	}
	// row[j+1] is the LCS length of the prev items seen so far and cur[:j+1].
	row := make([]int, len(cur)+1)
	for _, p := range prev {
		diag := 0
		for j, c := range cur {
			up := row[j+1]
			if p == c {
				row[j+1] = diag + 1
			} else if row[j] > up {
				row[j+1] = row[j]
			}
			diag = up
		}
	}
	return 1 - float64(row[len(cur)])/float64(len(cur))
}
