package llm

import (
	"context"
	"sync"
	"testing"
)

type stubEmbedder struct {
	id int
}

func (s *stubEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return []float32{float32(s.id)}, nil
}

func (s *stubEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{float32(s.id)}
	}
	return out, nil
}

func TestEmbedderHandle_NilInitialAndRoundTrip(t *testing.T) {
	h := NewEmbedderHandle(nil)
	if got := h.Get(); got != nil {
		t.Fatalf("expected nil initial embedder, got %T", got)
	}

	e := &stubEmbedder{id: 7}
	h.Set(e)
	if got := h.Get(); got != e {
		t.Fatalf("expected Set value to round-trip, got %v", got)
	}

	h.Set(nil)
	if got := h.Get(); got != nil {
		t.Fatalf("expected Set(nil) to clear, got %v", got)
	}
}

func TestEmbedderHandle_ConcurrentSetGet(t *testing.T) {
	h := NewEmbedderHandle(&stubEmbedder{id: 0})

	const workers = 16
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(workers * 2)

	for i := range workers {
		go func(id int) {
			defer wg.Done()
			for j := range iters {
				h.Set(&stubEmbedder{id: id*1000 + j})
			}
		}(i)
		go func() {
			defer wg.Done()
			for range iters {
				_ = h.Get()
			}
		}()
	}
	wg.Wait()

	if got := h.Get(); got == nil {
		t.Fatal("expected non-nil embedder after concurrent Set/Get")
	}
}
