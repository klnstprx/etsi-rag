// Package testutil provides shared test doubles used across multiple packages.
package testutil

import (
	"context"
	"strings"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/markers"
)

// FakeChunker is a deterministic chunker that emits the input as a single
// chunk per call. Image markers are stripped from content and surfaced on
// ImageRefs — matching the production chunker's contract — so marker-aware
// tests still hold. Used by tests that need *a* chunker to satisfy a
// constructor but don't care about the splitting strategy.
type FakeChunker struct{}

// Chunk implements chunking.Chunker.
func (FakeChunker) Chunk(_ context.Context, content string) ([]chunking.Chunk, error) {
	if content == "" {
		return nil, nil
	}
	cleaned, boundaries := markers.ExtractBoundaries(content)
	cleaned = strings.TrimSpace(cleaned)
	if cleaned == "" {
		return nil, nil
	}
	chunk := chunking.Chunk{Content: cleaned, Index: 0}
	if refs := uniqueRefs(boundaries); len(refs) > 0 {
		chunk.ImageRefs = refs
	}
	return []chunking.Chunk{chunk}, nil
}

func uniqueRefs(boundaries []markers.Boundary) []string {
	if len(boundaries) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(boundaries))
	out := make([]string, 0, len(boundaries))
	for _, b := range boundaries {
		if _, dup := seen[b.Ref]; dup {
			continue
		}
		seen[b.Ref] = struct{}{}
		out = append(out, b.Ref)
	}
	return out
}

// MockCounter counts words as tokens for testing.
type MockCounter struct{}

func (m *MockCounter) Count(_ context.Context, text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	return len(strings.Fields(text)), nil
}

func (m *MockCounter) Tokenize(_ context.Context, text string) ([]int, error) {
	words := strings.Fields(text)
	tokens := make([]int, len(words))
	for i := range tokens {
		tokens[i] = i
	}
	return tokens, nil
}

func (m *MockCounter) Detokenize(_ context.Context, tokens []int) (string, error) {
	parts := make([]string, len(tokens))
	for i := range parts {
		parts[i] = "word"
	}
	return strings.Join(parts, " "), nil
}
