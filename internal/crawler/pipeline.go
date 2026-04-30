package crawler

import "sync/atomic"

// CrawlItem flows from crawl stage to parse stage.
type CrawlItem struct {
	URL         string
	ContentType string // "html" or "pdf"
	Body        []byte
	ObjKey      string
	Source      CrawlSourceRef
	HTMLParsed  *HTMLResult // non-nil for HTML (already parsed for BFS links)
	Screenshot  []byte      // full-page PNG when go-rod captured one (HTML only)
}

// CrawlSourceRef is a lightweight reference to a crawl source, avoiding domain import in pipeline types.
type CrawlSourceRef struct {
	ID       int64
	Category string
	Language string
}

// ParseItem flows from parse stage to embed stage.
type ParseItem struct {
	URL         string
	Title       string
	Text        string
	ContentHash string
	ContentType string
	Language    string // normalised primary subtag from <html lang>, "" falls back to Source.Language
	ObjKey      string
	Source      CrawlSourceRef
	// Artifacts is the set of image bytes (PDF page renders or HTML screenshot)
	// that the embed stage will store in RustFS and link via document_artifacts.
	// Each ref matches a [[image:ref]] marker embedded in Text.
	Artifacts []ParsedArtifact
}

// ParsedArtifact carries one rendered image from parse stage to embed stage.
type ParsedArtifact struct {
	Ref      string // "p1","p2",… or "full"
	Kind     string // "pdf_page" | "html_screenshot"
	MimeType string // "image/png"
	Data     []byte
}

// RunProgress holds atomic counters for live progress tracking across pipeline stages.
// All fields are safe for concurrent read/write.
type RunProgress struct {
	// Crawl stage
	Discovered atomic.Int64
	Fetched    atomic.Int64
	CrawlFail  atomic.Int64

	// Parse stage
	Parsed    atomic.Int64
	ParseFail atomic.Int64

	// Embed stage
	Ingested              atomic.Int64
	Skipped               atomic.Int64
	EmbedFail             atomic.Int64
	ConsecutiveEmbedFails atomic.Int64 // shared across all embed workers

	// Stage activity indicators (1 = active, 0 = idle)
	CrawlActive atomic.Int32
	ParseActive atomic.Int32
	EmbedActive atomic.Int32
}

// ProgressSnapshot is a plain-value copy of RunProgress, safe for serialization.
type ProgressSnapshot struct {
	Discovered int64
	Fetched    int64
	CrawlFail  int64

	Parsed    int64
	ParseFail int64

	Ingested  int64
	Skipped   int64
	EmbedFail int64

	CrawlActive bool
	ParseActive bool
	EmbedActive bool
}

// Snapshot returns a point-in-time copy of all counters.
func (p *RunProgress) Snapshot() ProgressSnapshot {
	return ProgressSnapshot{
		Discovered:  p.Discovered.Load(),
		Fetched:     p.Fetched.Load(),
		CrawlFail:   p.CrawlFail.Load(),
		Parsed:      p.Parsed.Load(),
		ParseFail:   p.ParseFail.Load(),
		Ingested:    p.Ingested.Load(),
		Skipped:     p.Skipped.Load(),
		EmbedFail:   p.EmbedFail.Load(),
		CrawlActive: p.CrawlActive.Load() > 0,
		ParseActive: p.ParseActive.Load() > 0,
		EmbedActive: p.EmbedActive.Load() > 0,
	}
}
