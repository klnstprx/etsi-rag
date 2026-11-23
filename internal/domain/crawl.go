package domain

import (
	"slices"
	"time"
)

// CrawlSource represents a seed URL and its configuration.
//
// Key is the operator-chosen slug that crawl_pages and crawl_runs reference,
// so a source's seed URL or display name can change without orphaning
// history. Description and Tags only drive admin grouping and filtering.
type CrawlSource struct {
	Key         string   `koanf:"key"`
	Name        string   `koanf:"name"`
	SeedURL     string   `koanf:"seed_url"`
	Description string   `koanf:"description"`
	Tags        []string `koanf:"tags"`
	Enabled     bool     `koanf:"enabled"`
}

// SourceKeysWithTag returns the keys of every source tagged with tag, or nil
// when tag is empty or no source carries it.
func SourceKeysWithTag(sources []CrawlSource, tag string) []string {
	if tag == "" {
		return nil
	}
	var out []string
	for _, s := range sources {
		if slices.Contains(s.Tags, tag) {
			out = append(out, s.Key)
		}
	}
	return out
}

// CrawlPageStatus represents the status of a crawled page.
type CrawlPageStatus string

const (
	CrawlPagePending CrawlPageStatus = "pending"
	CrawlPageFetched CrawlPageStatus = "fetched"
	CrawlPageFailed  CrawlPageStatus = "failed"
	CrawlPageStale   CrawlPageStatus = "stale"
)

// CrawlPage represents a discovered and potentially crawled page.
type CrawlPage struct {
	ID             int64
	SourceKey      string
	URL            string
	ContentType    string // "html" or "pdf"
	Title          string
	ContentHash    string  // SHA-256 of extracted text
	ObjectStoreKey string  // key in RustFS
	DocID          *string // FK to documents.id
	Status         CrawlPageStatus
	ErrorMessage   *string
	LastCrawledAt  *time.Time
	LastChangedAt  *time.Time
}

// CrawlRunStatus represents the status of a crawl run.
type CrawlRunStatus string

const (
	CrawlRunRunning   CrawlRunStatus = "running"
	CrawlRunCompleted CrawlRunStatus = "completed"
	CrawlRunFailed    CrawlRunStatus = "failed"
	CrawlRunCancelled CrawlRunStatus = "cancelled"
)

// CrawlRun represents a single execution of the crawler.
type CrawlRun struct {
	ID              int64
	SourceKey       *string // nil = all sources
	Status          CrawlRunStatus
	PagesDiscovered int
	PagesFetched    int
	PagesChanged    int
	PagesFailed     int
	ErrorMessage    *string
	StartedAt       time.Time
	FinishedAt      *time.Time
}
