// Package diag holds startup-time and runtime non-fatal issues so they can be
// surfaced on the admin dashboard rather than disappearing into log files.
//
// Two surfaces:
//   - Set/Clear: called from the call sites that already log non-fatal warnings,
//     in addition to (not instead of) the existing log line.
//   - Register/Recheck: a callback per issue ID that re-runs the same probe;
//     the admin "Re-check" button calls it. Issues whose underlying init step
//     is a one-shot at boot (object store init, embedding cache warm) are
//     marked RequiresRestart=true; Recheck skips them so the operator gets
//     honest feedback that a restart is needed.
package diag

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Severity is the rendered severity of an Issue. Higher value = louder.
type Severity int

const (
	// SeverityWarn is the default; rendered amber on the dashboard. Use it
	// for non-fatal degraded-but-functioning states.
	SeverityWarn Severity = iota + 1
	// SeverityError is rendered red. Use it for states that block a major
	// feature (e.g., agent disabled, embedding backend missing).
	SeverityError
)

// Issue is one entry in the registry. ID is the stable handle used to
// idempotently Set/Clear the same condition; Detail and At update on each
// new Set with the same ID.
type Issue struct {
	ID              string
	Component       string
	Severity        Severity
	Message         string
	Detail          string
	Hint            string
	RequiresRestart bool
	At              time.Time
}

// Recheck re-runs the probe for an issue ID. nil result → Clear; non-nil →
// updated Set. Closures typically capture the relevant subsystem pointer.
type Recheck func(ctx context.Context) error

// Registry is a goroutine-safe map of Issue keyed by ID, plus an optional
// recheck function per ID. The zero value is unusable — call New.
type Registry struct {
	mu       sync.RWMutex
	issues   map[string]Issue
	rechecks map[string]rechecker
}

// rechecker bundles a recheck function with the metadata needed to construct
// an updated Issue when the recheck returns an error. The Set call path
// passes a full Issue; the recheck call path knows only the ID, so the
// metadata for "what kind of issue does this recheck produce" lives next to
// the function.
type rechecker struct {
	fn       Recheck
	template Issue // Used when the recheck reports an error
}

// New constructs an empty registry.
func New() *Registry {
	return &Registry{
		issues:   make(map[string]Issue),
		rechecks: make(map[string]rechecker),
	}
}

// Set adds or replaces the issue with the given ID. At is set to time.Now if
// the caller didn't supply one.
func (r *Registry) Set(i Issue) {
	if i.At.IsZero() {
		i.At = time.Now()
	}
	r.mu.Lock()
	r.issues[i.ID] = i
	r.mu.Unlock()
}

// Clear removes the issue with the given ID. No-op when absent.
func (r *Registry) Clear(id string) {
	r.mu.Lock()
	delete(r.issues, id)
	r.mu.Unlock()
}

// Register attaches a recheck function for an issue ID. The template is the
// Issue shape used when the recheck reports an error (the recheck itself only
// returns an error, not a full Issue). The template's ID and Component should
// match the ID being registered.
func (r *Registry) Register(id string, template Issue, fn Recheck) {
	r.mu.Lock()
	r.rechecks[id] = rechecker{fn: fn, template: template}
	r.mu.Unlock()
}

// Recheck re-runs every registered recheck except those whose issue is
// currently RequiresRestart, then Sets or Clears each issue from the result.
// Rechecks run even when their issue is absent, so a passing probe stays clear.
func (r *Registry) Recheck(ctx context.Context) {
	r.mu.RLock()
	type job struct {
		id string
		rc rechecker
	}
	var jobs []job
	for id, rc := range r.rechecks {
		if existing, ok := r.issues[id]; ok && existing.RequiresRestart {
			continue
		}
		jobs = append(jobs, job{id: id, rc: rc})
	}
	r.mu.RUnlock()

	for _, j := range jobs {
		err := j.rc.fn(ctx)
		if err == nil {
			r.Clear(j.id)
			continue
		}
		issue := j.rc.template
		issue.ID = j.id
		issue.Detail = err.Error()
		issue.At = time.Now()
		r.Set(issue)
	}
}

// Snapshot returns the current issues sorted by severity descending, then by
// At ascending (oldest first within a severity tier).
func (r *Registry) Snapshot() []Issue {
	r.mu.RLock()
	out := make([]Issue, 0, len(r.issues))
	for _, i := range r.issues {
		out = append(out, i)
	}
	r.mu.RUnlock()

	sort.Slice(out, func(a, b int) bool {
		if out[a].Severity != out[b].Severity {
			return out[a].Severity > out[b].Severity
		}
		return out[a].At.Before(out[b].At)
	})
	return out
}
