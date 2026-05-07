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
	"etsi-rag/internal/domain"
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
}

// New constructs a Registry pointed at path. The snapshot starts empty until
// Load (or Replace) succeeds.
func New(path string, logger zerolog.Logger) *Registry {
	return &Registry{path: path, logger: logger}
}

// Path returns the file path the Registry reads from and writes to.
func (r *Registry) Path() string {
	return r.path
}

// Load reads the file at r.Path(), validates it, and swaps the in-memory
// snapshot. On error the previous snapshot is left untouched, so a corrupted
// file at boot does not silently zero out the source list.
func (r *Registry) Load() error {
	loaded, err := config.LoadSourcesFile(r.path)
	if err != nil {
		return fmt.Errorf("loading %s: %w", r.path, err)
	}
	r.mu.Lock()
	r.sources = loaded
	r.mu.Unlock()
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
	return nil
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
