package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// newTestClient creates a Client pointing at the given test server.
func newTestClient(url string) *Client {
	c := openai.NewClient(
		option.WithBaseURL(url),
		option.WithAPIKey("test"),
	)
	return &Client{
		client:         &c,
		embeddingModel: "test-embed",
		chatModel:      "test-chat",
		maxTokens:      100,
		temperature:    0.0,
	}
}

func TestEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "test-embed",
			"data": []map[string]any{
				{
					"object":    "embedding",
					"index":     0,
					"embedding": []float64{0.1, 0.2, 0.3},
				},
			},
			"usage": map[string]int{"prompt_tokens": 5, "total_tokens": 5},
		})
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	emb, err := client.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(emb) != 3 {
		t.Fatalf("expected 3 dims, got %d", len(emb))
	}
	if emb[0] < 0.09 || emb[0] > 0.11 {
		t.Errorf("expected ~0.1, got %f", emb[0])
	}
}

func TestEmbedBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "test-embed",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{1.0, 0.0}},
				{"object": "embedding", "index": 1, "embedding": []float64{0.0, 1.0}},
			},
			"usage": map[string]int{"prompt_tokens": 10, "total_tokens": 10},
		})
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	embs, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(embs) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(embs))
	}
}

func TestChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"model":   "test-chat",
			"choices": []map[string]any{
				{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello! How can I help?",
					},
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     10,
				"completion_tokens": 6,
				"total_tokens":      16,
			},
		})
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	resp, err := client.Chat(context.Background(), []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Hi"),
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}

	if resp != "Hello! How can I help?" {
		t.Errorf("unexpected response: %q", resp)
	}
}

func TestNewClient(t *testing.T) {
	client := NewClient(ClientConfig{
		BaseURL:        "http://localhost:8081/v1",
		EmbeddingModel: "embed",
		ChatModel:      "chat",
		MaxTokens:      512,
		Temperature:    0.7,
	})

	if client.embeddingModel != "embed" {
		t.Errorf("unexpected embedding model: %s", client.embeddingModel)
	}
	if client.chatModel != "chat" {
		t.Errorf("unexpected chat model: %s", client.chatModel)
	}
}

func TestFloat64sToFloat32s(t *testing.T) {
	input := []float64{1.0, 2.5, -3.0}
	output := float64sToFloat32s(input)
	if len(output) != 3 {
		t.Fatalf("expected 3, got %d", len(output))
	}
	if output[0] != 1.0 || output[1] != 2.5 || output[2] != -3.0 {
		t.Errorf("unexpected values: %v", output)
	}
}
