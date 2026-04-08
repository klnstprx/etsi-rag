// Package testutil provides shared test doubles used across multiple packages.
package testutil

import (
	"context"
	"strings"
)

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
