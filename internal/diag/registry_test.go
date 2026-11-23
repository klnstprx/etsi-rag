package diag_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"etsi-rag/internal/diag"
)

func TestRegistry_SetSnapshotClear(t *testing.T) {
	r := diag.New()
	r.Set(diag.Issue{ID: "a", Component: "x", Severity: diag.SeverityWarn, Message: "first"})
	r.Set(diag.Issue{ID: "b", Component: "y", Severity: diag.SeverityError, Message: "second"})

	got := r.Snapshot()
	if len(got) != 2 {
		t.Fatalf("Snapshot len = %d, want 2", len(got))
	}
	// Errors before warns, regardless of insertion order.
	if got[0].ID != "b" || got[1].ID != "a" {
		t.Errorf("sort order: %v %v, want b then a", got[0].ID, got[1].ID)
	}

	r.Clear("a")
	got = r.Snapshot()
	if len(got) != 1 || got[0].ID != "b" {
		t.Errorf("after Clear, snapshot = %+v, want [b]", got)
	}

	r.Clear("does-not-exist") // no-op
}

func TestRegistry_SetOverwritesByID(t *testing.T) {
	r := diag.New()
	r.Set(diag.Issue{ID: "a", Severity: diag.SeverityWarn, Detail: "v1"})
	t1, _ := find(r, "a")
	time.Sleep(2 * time.Millisecond)
	r.Set(diag.Issue{ID: "a", Severity: diag.SeverityWarn, Detail: "v2"})
	t2, _ := find(r, "a")

	if t2.Detail != "v2" {
		t.Errorf("Detail = %q, want v2", t2.Detail)
	}
	if !t2.At.After(t1.At) {
		t.Errorf("At did not advance: t1=%v t2=%v", t1.At, t2.At)
	}
}

func TestRegistry_SnapshotSortByAtWithinSeverity(t *testing.T) {
	r := diag.New()
	now := time.Now()
	r.Set(diag.Issue{ID: "old", Severity: diag.SeverityWarn, At: now})
	r.Set(diag.Issue{ID: "new", Severity: diag.SeverityWarn, At: now.Add(time.Minute)})

	got := r.Snapshot()
	if got[0].ID != "old" || got[1].ID != "new" {
		t.Errorf("sort order: %v %v, want old new (oldest first within severity)", got[0].ID, got[1].ID)
	}
}

func TestRegistry_RecheckClearsOnNil(t *testing.T) {
	r := diag.New()
	template := diag.Issue{ID: "probe", Component: "x", Severity: diag.SeverityWarn, Message: "down"}
	r.Set(template)
	r.Register("probe", template, func(ctx context.Context) error { return nil })

	r.Recheck(context.Background())
	if _, ok := find(r, "probe"); ok {
		t.Error("issue should have been cleared")
	}
}

func TestRegistry_RecheckSetsOnError(t *testing.T) {
	r := diag.New()
	template := diag.Issue{
		ID: "probe", Component: "x", Severity: diag.SeverityError,
		Message: "static message", Hint: "static hint",
	}
	probeErr := errors.New("dial tcp: refused")
	r.Register("probe", template, func(ctx context.Context) error { return probeErr })

	r.Recheck(context.Background())
	got, ok := find(r, "probe")
	if !ok {
		t.Fatal("issue should have been set")
	}
	if got.Message != "static message" || got.Hint != "static hint" {
		t.Errorf("template fields not preserved: %+v", got)
	}
	if got.Detail != probeErr.Error() {
		t.Errorf("Detail = %q, want %q", got.Detail, probeErr.Error())
	}
}

func TestRegistry_RecheckSkipsRequiresRestart(t *testing.T) {
	r := diag.New()
	called := false

	stuck := diag.Issue{
		ID: "objstore", Component: "objectstore", Severity: diag.SeverityError,
		Message: "down", RequiresRestart: true,
	}
	r.Set(stuck)
	r.Register("objstore", stuck, func(ctx context.Context) error {
		called = true
		return nil
	})

	r.Recheck(context.Background())
	if called {
		t.Error("RequiresRestart issue should not be rechecked")
	}
	if _, ok := find(r, "objstore"); !ok {
		t.Error("RequiresRestart issue should remain set")
	}
}

func TestRegistry_RecheckCoversAbsentIssues(t *testing.T) {
	// A recheck registered with no current issue must still run on Recheck()
	// — otherwise a transient condition that already cleared would never get
	// a "still passing" confirmation.
	r := diag.New()
	called := false
	r.Register("probe",
		diag.Issue{ID: "probe", Component: "x", Severity: diag.SeverityWarn, Message: "m"},
		func(ctx context.Context) error {
			called = true
			return nil
		})

	r.Recheck(context.Background())
	if !called {
		t.Error("recheck for absent issue should still run")
	}
}

func TestRegistry_ConcurrentSetRecheckSnapshot(t *testing.T) {
	r := diag.New()
	r.Register("probe",
		diag.Issue{ID: "probe", Component: "x", Severity: diag.SeverityWarn, Message: "m"},
		func(ctx context.Context) error { return nil })

	const N = 50
	var wg sync.WaitGroup
	wg.Add(3 * N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			r.Set(diag.Issue{ID: "probe", Severity: diag.SeverityWarn, Message: "m"})
		}()
		go func() {
			defer wg.Done()
			_ = r.Snapshot()
		}()
		go func() {
			defer wg.Done()
			r.Recheck(context.Background())
		}()
	}
	wg.Wait()
}

// find looks an issue up through Snapshot, the registry's only read path.
func find(r *diag.Registry, id string) (diag.Issue, bool) {
	for _, i := range r.Snapshot() {
		if i.ID == id {
			return i, true
		}
	}
	return diag.Issue{}, false
}
