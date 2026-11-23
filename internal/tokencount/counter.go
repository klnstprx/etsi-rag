// Package tokencount provides token counting for different LLM backends.
package tokencount

import (
	"context"
	"errors"
)

// ErrBackend marks a tokenizer that could not be reached or kept failing with
// 5xx after its retries, as opposed to one rejecting a particular text. Check
// with errors.Is.
var ErrBackend = errors.New("tokenizer backend unavailable")

// Counter counts tokens in text using a backend-specific tokenizer.
type Counter interface {
	Count(ctx context.Context, text string) (int, error)
	Tokenize(ctx context.Context, text string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
}
