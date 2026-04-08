// Package llm provides the LLM client using the official openai-go SDK.
// llama-server is OpenAI API compliant, so we use the SDK directly.
package llm

import "context"

// Embedder defines the interface for embedding operations.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}
