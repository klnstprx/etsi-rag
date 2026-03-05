package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// Compile-time check that Client implements LLM.
var _ LLM = (*Client)(nil)

// ClientConfig holds everything needed to construct an LLM client.
type ClientConfig struct {
	BaseURL        string
	APIKey         string
	EmbeddingModel string
	ChatModel      string
	MaxTokens      int64
	Temperature    float64
}

// Client wraps the OpenAI client for embedding and chat operations.
type Client struct {
	client         *openai.Client
	embeddingModel string
	chatModel      string
	maxTokens      int64
	temperature    float64
}

// NewClient creates a new LLM client from a ClientConfig.
func NewClient(cfg ClientConfig) *Client {
	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = "not-needed"
	}

	client := openai.NewClient(
		option.WithBaseURL(cfg.BaseURL),
		option.WithAPIKey(apiKey),
	)

	return &Client{
		client:         &client,
		embeddingModel: cfg.EmbeddingModel,
		chatModel:      cfg.ChatModel,
		maxTokens:      cfg.MaxTokens,
		temperature:    cfg.Temperature,
	}
}

// Embed generates an embedding for a single text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := c.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(text),
		},
		Model: openai.EmbeddingModel(c.embeddingModel),
	})
	if err != nil {
		return nil, fmt.Errorf("creating embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	return float64sToFloat32s(resp.Data[0].Embedding), nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := c.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: texts,
		},
		Model: openai.EmbeddingModel(c.embeddingModel),
	})
	if err != nil {
		return nil, fmt.Errorf("creating batch embeddings: %w", err)
	}

	result := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		result[i] = float64sToFloat32s(d.Embedding)
	}

	return result, nil
}

// Chat generates a response from a conversation.
func (c *Client) Chat(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	resp, err := c.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       shared.ChatModel(c.chatModel),
		MaxTokens:   openai.Int(c.maxTokens),
		Temperature: openai.Float(c.temperature),
	})
	if err != nil {
		return "", fmt.Errorf("chat completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}

	return resp.Choices[0].Message.Content, nil
}

// ChatStream generates a streaming response, sending tokens to a channel.
func (c *Client) ChatStream(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (<-chan string, <-chan error) {
	tokens := make(chan string, 64)
	errc := make(chan error, 1)

	go func() {
		defer close(tokens)
		defer close(errc)

		stream := c.client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Messages:    messages,
			Model:       shared.ChatModel(c.chatModel),
			MaxTokens:   openai.Int(c.maxTokens),
			Temperature: openai.Float(c.temperature),
		})

		for stream.Next() {
			chunk := stream.Current()
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					tokens <- choice.Delta.Content
				}
			}
		}

		if err := stream.Err(); err != nil {
			errc <- fmt.Errorf("streaming chat: %w", err)
		}
	}()

	return tokens, errc
}

// ChatWithTools generates a response that may include tool calls.
func (c *Client) ChatWithTools(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) (*openai.ChatCompletion, error) {
	resp, err := c.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       shared.ChatModel(c.chatModel),
		MaxTokens:   openai.Int(c.maxTokens),
		Temperature: openai.Float(c.temperature),
		Tools:       tools,
	})
	if err != nil {
		return nil, fmt.Errorf("chat with tools: %w", err)
	}
	return resp, nil
}

// float64sToFloat32s converts a []float64 to []float32.
func float64sToFloat32s(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
