package agent

import (
	"sync"
	"testing"
)

func TestHandle_NilInitialAndRoundTrip(t *testing.T) {
	h := NewHandle(nil)
	if got := h.Get(); got != nil {
		t.Fatalf("expected nil initial agent, got %v", got)
	}

	a := &Agent{}
	h.Set(a)
	if got := h.Get(); got != a {
		t.Fatalf("expected Set value to round-trip, got %v", got)
	}

	h.Set(nil)
	if got := h.Get(); got != nil {
		t.Fatalf("expected Set(nil) to clear, got %v", got)
	}
}

func TestHandle_ConcurrentSetGet(t *testing.T) {
	h := NewHandle(&Agent{})

	const workers = 16
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(workers * 2)

	for range workers {
		go func() {
			defer wg.Done()
			for range iters {
				h.Set(&Agent{})
			}
		}()
		go func() {
			defer wg.Done()
			for range iters {
				_ = h.Get()
			}
		}()
	}
	wg.Wait()

	if got := h.Get(); got == nil {
		t.Fatal("expected non-nil agent after concurrent Set/Get")
	}
}
