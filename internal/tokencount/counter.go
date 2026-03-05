// Package tokencount provides token counting for different LLM backends.
package tokencount

import "context"

// Counter counts tokens in text using a backend-specific tokenizer.
type Counter interface {
	Count(ctx context.Context, text string) (int, error)
	Tokenize(ctx context.Context, text string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
}
