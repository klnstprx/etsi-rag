package llm

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Compile-time check.
var _ Embedder = (*EmbedClient)(nil)

// DefaultMaxRetries is the fallback retry count for transient 5xx responses
// when a backend doesn't override it. ~120s exponential-backoff budget at
// openai-go's default cadence — long enough to outlast a llama-server restart
// with cold model reload.
const DefaultMaxRetries = 10

// resolveMaxRetries normalizes a configured retry count: 0 means "use default",
// negative is treated as 0 retries (i.e. one attempt, no retries).
func resolveMaxRetries(n int) int {
	switch {
	case n == 0:
		return DefaultMaxRetries
	case n < 0:
		return 0
	default:
		return n
	}
}

// defaultBatchSize is the embedding sub-batch size when the config leaves it unset.
const defaultBatchSize = 32

// defaultEmbedRequestTimeout bounds each embedding request attempt when the
// config leaves it unset, so a stalled backend fails the call instead of
// hanging a crawl. It is far above a full sub-batch on a local llama-server.
const defaultEmbedRequestTimeout = 2 * time.Minute

// newOpenAI builds an SDK client for an OpenAI-compatible backend. llama.cpp
// ignores the API key, but the SDK requires one to be set.
func newOpenAI(baseURL, apiKey string, opts ...option.RequestOption) *openai.Client {
	if apiKey == "" {
		apiKey = "not-needed"
	}
	c := openai.NewClient(append([]option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}, opts...)...)
	return &c
}

// EmbedClientConfig holds everything needed to construct an embedding client.
type EmbedClientConfig struct {
	BaseURL        string
	APIKey         string
	Model          string
	BatchSize      int
	MaxRetries     int
	RequestTimeout time.Duration // per attempt; 0 = defaultEmbedRequestTimeout
}

// EmbedClient wraps the OpenAI client for embedding operations.
type EmbedClient struct {
	client    *openai.Client
	model     string
	batchSize int
}

// NewEmbedClient creates a new embedding client from an EmbedClientConfig.
//
// Retries on transient upstream 5xx (e.g. llama-server crash
// restart) absorb backend flakiness so the crawler's consecutive-fail circuit
// breaker doesn't trip on a recoverable outage. openai-go uses exponential
// backoff with jitter. See DefaultMaxRetries for the fallback budget.
func NewEmbedClient(cfg EmbedClientConfig) *EmbedClient {
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}
	timeout := cfg.RequestTimeout
	if timeout <= 0 {
		timeout = defaultEmbedRequestTimeout
	}
	return &EmbedClient{
		client: newOpenAI(cfg.BaseURL, cfg.APIKey,
			option.WithMaxRetries(resolveMaxRetries(cfg.MaxRetries)),
			option.WithRequestTimeout(timeout),
		),
		model:     cfg.Model,
		batchSize: batchSize,
	}
}

// Embed generates an embedding for a single text.
func (c *EmbedClient) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := c.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(text),
		},
		Model: c.model,
	})
	if err != nil {
		return nil, fmt.Errorf("creating embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, ErrEmptyEmbedding
	}

	return float64sToFloat32s(resp.Data[0].Embedding), nil
}

// EmbedBatch generates embeddings for multiple texts, splitting into
// sub-batches of batchSize to limit memory pressure on the server.
func (c *EmbedClient) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, 0, len(texts))

	for i := 0; i < len(texts); i += c.batchSize {
		end := min(i+c.batchSize, len(texts))
		batch := texts[i:end]

		resp, err := c.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{
				OfArrayOfStrings: batch,
			},
			Model: c.model,
		})
		if err != nil {
			return nil, fmt.Errorf("creating batch embeddings: %w", err)
		}
		if len(resp.Data) != len(batch) {
			// A short response would silently misalign chunks and embeddings
			// downstream; treat any shortfall as an empty-response failure.
			return nil, ErrEmptyEmbedding
		}

		// Sort by Index to guarantee input-order alignment regardless of server response order.
		sort.Slice(resp.Data, func(a, b int) bool {
			return resp.Data[a].Index < resp.Data[b].Index
		})

		for _, d := range resp.Data {
			result = append(result, float64sToFloat32s(d.Embedding))
		}
	}

	return result, nil
}

// ChatClientConfig holds everything needed to construct a chat client.
type ChatClientConfig struct {
	BaseURL     string
	APIKey      string
	Model       string
	MaxTokens   int64
	Temperature float64
	MaxRetries  int
}

// ChatClient wraps the OpenAI client for chat operations.
type ChatClient struct {
	client      *openai.Client
	model       string
	maxTokens   int64
	temperature float64
}

// NewChatClient creates a new chat client from a ChatClientConfig.
//
// See NewEmbedClient for retry rationale.
func NewChatClient(cfg ChatClientConfig) *ChatClient {
	return &ChatClient{
		client:      newOpenAI(cfg.BaseURL, cfg.APIKey, option.WithMaxRetries(resolveMaxRetries(cfg.MaxRetries))),
		model:       cfg.Model,
		maxTokens:   cfg.MaxTokens,
		temperature: cfg.Temperature,
	}
}

// params builds the request shared by the blocking and streaming calls, so
// both send the same model, sampling settings and tools.
func (c *ChatClient) params(messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages:            messages,
		Model:               c.model,
		MaxCompletionTokens: openai.Int(c.maxTokens),
		Temperature:         openai.Float(c.temperature),
		Tools:               tools,
	}
}

// ChatWithTools generates a response that may include tool calls.
func (c *ChatClient) ChatWithTools(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) (*openai.ChatCompletion, error) {
	resp, err := c.client.Chat.Completions.New(ctx, c.params(messages, tools))
	if err != nil {
		return nil, fmt.Errorf("chat with tools: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, ErrEmptyCompletion
	}
	return resp, nil
}

// ChatStreamResult carries the assembled completion (or terminal error) from a
// streaming chat call. Exactly one value is sent on the channel returned by
// ChatStreamWithTools, after which the channel is closed.
type ChatStreamResult struct {
	Completion *openai.ChatCompletion
	Err        error
}

// ChatStreamWithTools is the streaming counterpart to ChatWithTools. Content
// deltas are forwarded to the returned tokens channel as they arrive; once the
// upstream stream finishes, the accumulated *openai.ChatCompletion (with
// merged tool_calls) is sent on the result channel. Both channels are closed
// by this method's goroutine.
func (c *ChatClient) ChatStreamWithTools(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolParam,
) (<-chan string, <-chan ChatStreamResult) {
	tokens := make(chan string, 1024)
	resultCh := make(chan ChatStreamResult, 1)

	go func() {
		defer close(tokens)
		defer close(resultCh)

		acc := openai.ChatCompletionAccumulator{}
		stream := c.client.Chat.Completions.NewStreaming(ctx, c.params(messages, tools))

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					tokens <- choice.Delta.Content
				}
			}
		}

		res := ChatStreamResult{Completion: &acc.ChatCompletion}
		if err := stream.Err(); err != nil {
			res.Err = fmt.Errorf("streaming chat with tools: %w", err)
		}
		resultCh <- res
	}()

	return tokens, resultCh
}

// float64sToFloat32s converts a []float64 to []float32.
func float64sToFloat32s(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
