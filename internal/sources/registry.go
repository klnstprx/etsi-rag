// Package sources owns the in-memory snapshot of crawl sources backed by
// config/sources.toml. The Registry is the single source of truth at runtime:
// the admin handler edits it (which validates + persists), the crawler reads
// it on each run, and the agent's list_sources tool reads it on each turn.
//
// Validation lives in internal/config — this package just composes Load and
// Write into a thread-safe, validating, atomic-replace surface.
package sources

import (
	"fmt"
	"sync"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
	"etsi-rag/internal/diag"
	"etsi-rag/internal/domain"
)

// Diag issue IDs surfaced by this package. Exported so the admin dashboard
// and tests can refer to them without re-typing the strings.
const (
	IssueParse = "sources.parse"
	IssueEmpty = "sources.empty"
)

// Registry holds the canonical in-memory snapshot of crawl sources.
//
// Concurrent Snapshot/Replace is safe; readers see a consistent slice (a
// defensive copy on every read) and writers swap the snapshot atomically
// after the on-disk write succeeds.
type Registry struct {
	mu      sync.RWMutex
	path    string
	sources []domain.CrawlSource
	logger  zerolog.Logger
	diag    *diag.Registry // optional; nil = no diagnostic surfacing
}

// New constructs a Registry pointed at path. The snapshot starts empty until
// Load (or Replace) succeeds. Diag is optional: when set, Load and Replace
// publish IssueParse/IssueEmpty so the admin dashboard can surface them.
func New(path string, logger zerolog.Logger) *Registry {
	return &Registry{path: path, logger: logger}
}

// SetDiag attaches a diag registry. Pass nil to detach.
func (r *Registry) SetDiag(d *diag.Registry) {
	r.mu.Lock()
	r.diag = d
	r.mu.Unlock()
}

// Path returns the file path the Registry reads from and writes to.
func (r *Registry) Path() string {
	return r.path
}

// Load reads the file at r.Path(), validates it, and swaps the in-memory
// snapshot. On error the previous snapshot is left untouched, so a corrupted
// file at boot does not silently zero out the source list.
//
// Publishes IssueParse on parse/validation failure and IssueEmpty when the
// file successfully loads but contains zero sources.
func (r *Registry) Load() error {
	loaded, err := config.LoadSourcesFile(r.path)
	if err != nil {
		r.publishParse(err)
		return fmt.Errorf("loading %s: %w", r.path, err)
	}
	r.mu.Lock()
	r.sources = loaded
	r.mu.Unlock()
	r.publishParse(nil)
	r.publishEmpty(len(loaded) == 0)
	return nil
}

// Snapshot returns a defensive copy of the current source list. Callers may
// mutate the returned slice freely without affecting the Registry.
func (r *Registry) Snapshot() []domain.CrawlSource {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]domain.CrawlSource(nil), r.sources...)
}

// Replace validates the given sources, writes them to disk atomically, and on
// success swaps the in-memory snapshot. If validation or the write fails, the
// snapshot AND the on-disk file are left untouched (config.WriteSources writes
// to a temp file first, then renames).
//
// On success, publishes a fresh "all good" diagnostic state (clears IssueParse
// and updates IssueEmpty based on the new length).
func (r *Registry) Replace(sources []domain.CrawlSource) error {
	if err := config.WriteSources(sources, r.path); err != nil {
		return err
	}
	// Defensive copy on the way in so a caller mutating the slice after
	// Replace returns cannot retroactively change the Registry's view.
	stored := append([]domain.CrawlSource(nil), sources...)
	r.mu.Lock()
	r.sources = stored
	r.mu.Unlock()
	r.publishParse(nil)
	r.publishEmpty(len(stored) == 0)
	return nil
}

// publishParse sets or clears IssueParse on the attached diag registry. nil
// err = clear; non-nil = set.
func (r *Registry) publishParse(err error) {
	r.mu.RLock()
	d := r.diag
	r.mu.RUnlock()
	if d == nil {
		return
	}
	if err == nil {
		d.Clear(IssueParse)
		return
	}
	d.Set(diag.Issue{
		ID:        IssueParse,
		Component: "sources",
		Severity:  diag.SeverityError,
		Message:   "sources.toml failed to load",
		Detail:    err.Error(),
		Hint:      "Edit sources.toml and reload, or fix at /admin/crawler",
	})
}

// publishEmpty sets or clears IssueEmpty depending on whether the in-memory
// snapshot is empty.
func (r *Registry) publishEmpty(isEmpty bool) {
	r.mu.RLock()
	d := r.diag
	r.mu.RUnlock()
	if d == nil {
		return
	}
	if !isEmpty {
		d.Clear(IssueEmpty)
		return
	}
	d.Set(diag.Issue{
		ID:        IssueEmpty,
		Component: "sources",
		Severity:  diag.SeverityWarn,
		Message:   "No crawl sources configured",
		Hint:      "Add a source at /admin/crawler",
	})
}

// KnownKeys returns the keys of every source currently in the snapshot, used
// by the boot-time orphan-page reconciliation in server startup.
func (r *Registry) KnownKeys() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	keys := make([]string, 0, len(r.sources))
	for _, s := range r.sources {
		keys = append(keys, s.Key)
	}
	return keys
}
