package llm

import "errors"

// Sentinel errors for empty upstream responses. They let HTTP handlers
// distinguish "the model server answered but gave us nothing usable" (a 502
// Bad Gateway condition) from a genuine internal fault, via errors.Is. They
// carry no extra context, so callers return them directly rather than wrapping.
var (
	// ErrEmptyEmbedding is returned when the embedding endpoint responds
	// successfully but with no embedding data.
	ErrEmptyEmbedding = errors.New("embedding service returned no data")

	// ErrEmptyCompletion is returned when the chat endpoint responds
	// successfully but with no completion choices.
	ErrEmptyCompletion = errors.New("chat service returned no completion choices")
)
