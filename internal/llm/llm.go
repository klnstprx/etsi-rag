// Package llm provides the LLM client using the official openai-go SDK.
// llama-server is OpenAI API compliant, so we use the SDK directly.
package llm

import (
	"context"

	"github.com/openai/openai-go"
)

// LLM defines the interface for embedding and chat operations.
type LLM interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	Chat(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error)
	ChatStream(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (<-chan string, <-chan error)
}
