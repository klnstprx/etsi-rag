package domain

import "time"

// CrawlSource represents a seed URL and its configuration.
//
// Key is a stable slug (e.g. "grado-informatica") chosen by the operator in
// sources.toml. It is the foreign key used by crawl_pages and crawl_runs, so
// a source's seed_url or display name can change without orphaning history.
//
// Description and Tags are agent-facing scoping hints surfaced through the
// list_sources tool. They are not persisted on documents and not used as SQL
// filters; the agent reads them and decides whether to scope a search by
// passing source_keys to semantic_search.
type CrawlSource struct {
	Key         string   `koanf:"key"`
	Name        string   `koanf:"name"`
	SeedURL     string   `koanf:"seed_url"`
	Description string   `koanf:"description"`
	Tags        []string `koanf:"tags"`
	Enabled     bool     `koanf:"enabled"`
}

// SourceKeysWithAnyTag returns the keys of every source whose Tags slice
// contains at least one of `tags` (OR semantics). Empty-string tags are
// ignored. Returns nil when no tags are supplied or no source matches.
//
// Shared by the admin Documents filter (single-tag intersection downstream)
// and the agent's semantic_search scope resolver (multi-tag union upstream)
// — both walk the same primitive: source.Tags membership.
func SourceKeysWithAnyTag(sources []CrawlSource, tags ...string) []string {
	if len(tags) == 0 {
		return nil
	}
	want := make(map[string]struct{}, len(tags))
	for _, t := range tags {
		if t == "" {
			continue
		}
		want[t] = struct{}{}
	}
	if len(want) == 0 {
		return nil
	}
	var out []string
	for _, s := range sources {
		for _, st := range s.Tags {
			if _, ok := want[st]; ok {
				out = append(out, s.Key)
				break
			}
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
