package domain

import "time"

// CrawlSource represents a seed URL and its configuration.
type CrawlSource struct {
	ID       int64
	Name     string
	SeedURL  string
	Category string
	Language string
	Enabled  bool
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
	SourceID       int64
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
	SourceID        *int64 // nil = all sources
	Status          CrawlRunStatus
	PagesDiscovered int
	PagesFetched    int
	PagesChanged    int
	PagesFailed     int
	ErrorMessage    *string
	StartedAt       time.Time
	FinishedAt      *time.Time
}
