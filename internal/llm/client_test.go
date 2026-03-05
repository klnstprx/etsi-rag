package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
		client:      &c,
		model:       "test-model",
		maxTokens:   100,
		temperature: 0.5,
	}
}

// chatResponse builds a raw chat completion JSON response.
func chatResponse(content string) map[string]any {
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
				},
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	}
}

// embeddingResponse builds a raw embedding JSON response.
func embeddingResponse(embeddings ...[]float64) map[string]any {
	data := make([]map[string]any, len(embeddings))
	for i, emb := range embeddings {
		data[i] = map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": emb,
		}
	}
	return map[string]any{
		"object": "list",
		"model":  "test-model",
		"data":   data,
		"usage":  map[string]int{"prompt_tokens": 5, "total_tokens": 5},
	}
}

// sseChunk formats a single SSE data line for streaming.
func sseChunk(content string, finishReason *string) string {
	fr := "null"
	if finishReason != nil {
		fr = fmt.Sprintf("%q", *finishReason)
	}
	return fmt.Sprintf("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1234567890,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":%s}]}\n\n", content, fr)
}

func ptr(s string) *string { return &s }

// --- NewClient ---

func TestNewClient(t *testing.T) {
	tests := []struct {
		name      string
		cfg       ClientConfig
		wantModel string
		wantMax   int64
		wantTemp  float64
	}{
		{
			name: "all fields",
			cfg: ClientConfig{
				BaseURL:     "http://localhost:8081/v1",
				APIKey:      "sk-test",
				Model:       "llama3",
				MaxTokens:   512,
				Temperature: 0.7,
			},
			wantModel: "llama3",
			wantMax:   512,
			wantTemp:  0.7,
		},
		{
			name: "empty api key gets default",
			cfg: ClientConfig{
				BaseURL: "http://localhost:8081/v1",
				Model:   "chat",
			},
			wantModel: "chat",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewClient(tt.cfg)
			if c.model != tt.wantModel {
				t.Errorf("model = %q, want %q", c.model, tt.wantModel)
			}
			if c.maxTokens != tt.wantMax {
				t.Errorf("maxTokens = %d, want %d", c.maxTokens, tt.wantMax)
			}
			if c.temperature != tt.wantTemp {
				t.Errorf("temperature = %f, want %f", c.temperature, tt.wantTemp)
			}
			if c.client == nil {
				t.Error("client is nil")
			}
		})
	}
}

// --- Embed ---

func TestEmbed(t *testing.T) {
	tests := []struct {
		name      string
		handler   http.HandlerFunc
		wantLen   int
		wantFirst float32
		wantErr   bool
	}{
		{
			name: "success",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(embeddingResponse([]float64{0.1, 0.2, 0.3}))
			},
			wantLen:   3,
			wantFirst: 0.1,
		},
		{
			name: "empty data",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"object": "list",
					"model":  "test",
					"data":   []any{},
					"usage":  map[string]int{"prompt_tokens": 0, "total_tokens": 0},
				})
			},
			wantErr: true,
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "model loading", "type": "server_error"},
				})
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			client := newTestClient(srv.URL)
			emb, err := client.Embed(context.Background(), "hello")

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(emb) != tt.wantLen {
				t.Fatalf("len = %d, want %d", len(emb), tt.wantLen)
			}
			if emb[0] != tt.wantFirst {
				t.Errorf("first = %f, want %f", emb[0], tt.wantFirst)
			}
		})
	}
}

func TestEmbed_RequestBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "test-model" {
			t.Errorf("model = %v, want test-model", req["model"])
		}
		if req["input"] != "test text" {
			t.Errorf("input = %v, want test text", req["input"])
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s, want /embeddings", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embeddingResponse([]float64{1.0}))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	_, err := client.Embed(context.Background(), "test text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- EmbedBatch ---

func TestEmbedBatch(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		texts   []string
		wantLen int
		wantErr bool
	}{
		{
			name: "success",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(embeddingResponse(
					[]float64{1.0, 0.0},
					[]float64{0.0, 1.0},
				))
			},
			texts:   []string{"a", "b"},
			wantLen: 2,
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
			},
			texts:   []string{"a"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			client := newTestClient(srv.URL)
			embs, err := client.EmbedBatch(context.Background(), tt.texts)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(embs) != tt.wantLen {
				t.Fatalf("len = %d, want %d", len(embs), tt.wantLen)
			}
		})
	}
}

func TestEmbedBatch_Values(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embeddingResponse(
			[]float64{0.5, -0.5},
			[]float64{1.0, 0.0},
		))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	embs, err := client.EmbedBatch(context.Background(), []string{"x", "y"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if embs[0][0] != 0.5 || embs[0][1] != -0.5 {
		t.Errorf("embs[0] = %v, want [0.5, -0.5]", embs[0])
	}
	if embs[1][0] != 1.0 || embs[1][1] != 0.0 {
		t.Errorf("embs[1] = %v, want [1.0, 0.0]", embs[1])
	}
}

func TestEmbedBatch_RequestBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		input, ok := req["input"].([]any)
		if !ok {
			t.Fatalf("input is not array: %T", req["input"])
		}
		if len(input) != 3 {
			t.Errorf("input len = %d, want 3", len(input))
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embeddingResponse(
			[]float64{1.0}, []float64{2.0}, []float64{3.0},
		))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	_, err := client.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedBatch_SubBatching(t *testing.T) {
	var callCount int
	var batchSizes []int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		input := req["input"].([]any)
		batchSizes = append(batchSizes, len(input))

		embs := make([][]float64, len(input))
		for i := range input {
			embs[i] = []float64{float64(i) + 1.0}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embeddingResponse(embs...))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	client.embedBatchSize = 2

	texts := []string{"a", "b", "c", "d", "e"}
	embs, err := client.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(embs) != 5 {
		t.Fatalf("got %d embeddings, want 5", len(embs))
	}

	// With batch size 2 and 5 texts: expect 3 API calls (2+2+1)
	if callCount != 3 {
		t.Errorf("API calls = %d, want 3", callCount)
	}

	wantSizes := []int{2, 2, 1}
	if len(batchSizes) != len(wantSizes) {
		t.Fatalf("batch sizes = %v, want %v", batchSizes, wantSizes)
	}
	for i, want := range wantSizes {
		if batchSizes[i] != want {
			t.Errorf("batch %d size = %d, want %d", i, batchSizes[i], want)
		}
	}

	// Verify embeddings are correctly stitched in order
	if embs[0][0] != 1.0 {
		t.Errorf("embs[0] = %v, want [1.0]", embs[0])
	}
	if embs[2][0] != 1.0 {
		// Third text is first in third batch → value 1.0
		t.Errorf("embs[2] = %v, want [1.0]", embs[2])
	}
}

func TestEmbedBatch_SubBatchSingleBatch(t *testing.T) {
	var callCount int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		input := req["input"].([]any)

		embs := make([][]float64, len(input))
		for i := range input {
			embs[i] = []float64{1.0}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embeddingResponse(embs...))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	client.embedBatchSize = 10

	// 3 texts with batch size 10 → single API call
	embs, err := client.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount != 1 {
		t.Errorf("API calls = %d, want 1", callCount)
	}
	if len(embs) != 3 {
		t.Errorf("got %d embeddings, want 3", len(embs))
	}
}

// --- Chat ---

func TestChat(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		want    string
		wantErr bool
	}{
		{
			name: "success",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(chatResponse("Hello!"))
			},
			want: "Hello!",
		},
		{
			name: "empty choices",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"id":      "chatcmpl-test",
					"object":  "chat.completion",
					"created": 1234567890,
					"model":   "test",
					"choices": []any{},
					"usage":   map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
				})
			},
			wantErr: true,
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "overloaded", "type": "server_error"},
				})
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			client := newTestClient(srv.URL)
			msgs := []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hi"),
					},
				}},
			}

			resp, err := client.Chat(context.Background(), msgs)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp != tt.want {
				t.Errorf("response = %q, want %q", resp, tt.want)
			}
		})
	}
}

func TestChat_RequestBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "test-model" {
			t.Errorf("model = %v, want test-model", req["model"])
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s, want /chat/completions", r.URL.Path)
		}
		// max_tokens should be 100 (from newTestClient)
		if mt, ok := req["max_tokens"].(float64); !ok || int64(mt) != 100 {
			t.Errorf("max_tokens = %v, want 100", req["max_tokens"])
		}
		// temperature should be 0.5
		if temp, ok := req["temperature"].(float64); !ok || temp != 0.5 {
			t.Errorf("temperature = %v, want 0.5", req["temperature"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatResponse("ok"))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	msgs := []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String("test"),
			},
		}},
	}
	_, err := client.Chat(context.Background(), msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- ChatStream ---

func TestChatStream(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		want    string
		wantErr bool
	}{
		{
			name: "multiple chunks",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				flusher := w.(http.Flusher)

				fmt.Fprint(w, sseChunk("Hello", nil))
				flusher.Flush()
				fmt.Fprint(w, sseChunk(" world", nil))
				flusher.Flush()
				fmt.Fprint(w, sseChunk("", ptr("stop")))
				flusher.Flush()
				fmt.Fprint(w, "data: [DONE]\n\n")
				flusher.Flush()
			},
			want: "Hello world",
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "fail", "type": "server_error"},
				})
			},
			wantErr: true,
		},
		{
			name: "empty stream",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: [DONE]\n\n")
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			client := newTestClient(srv.URL)
			msgs := []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hi"),
					},
				}},
			}

			tokens, errc := client.ChatStream(context.Background(), msgs)

			var result string
			for tok := range tokens {
				result += tok
			}

			if err := <-errc; err != nil {
				if !tt.wantErr {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if tt.wantErr {
				t.Fatal("expected error, got nil")
			}
			if result != tt.want {
				t.Errorf("result = %q, want %q", result, tt.want)
			}
		})
	}
}

func TestChatStream_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		// Send one chunk, then wait (simulating a slow server)
		fmt.Fprint(w, sseChunk("start", nil))
		flusher.Flush()

		// Block until the client disconnects
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	client := newTestClient(srv.URL)
	msgs := []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String("Hi"),
			},
		}},
	}

	tokens, errc := client.ChatStream(ctx, msgs)

	// Read first token then cancel
	tok := <-tokens
	if tok != "start" {
		t.Errorf("first token = %q, want %q", tok, "start")
	}
	cancel()

	// Drain remaining tokens
	for range tokens {
	}

	// Should get a context-related error
	if err := <-errc; err == nil {
		t.Error("expected error after cancellation, got nil")
	}
}

// --- ChatWithTools ---

func TestChatWithTools(t *testing.T) {
	tests := []struct {
		name          string
		handler       http.HandlerFunc
		wantToolCalls int
		wantContent   string
		wantErr       bool
	}{
		{
			name: "tool call response",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"id":      "chatcmpl-tools",
					"object":  "chat.completion",
					"created": 1234567890,
					"model":   "test-model",
					"choices": []map[string]any{
						{
							"index":         0,
							"finish_reason": "tool_calls",
							"message": map[string]any{
								"role":    "assistant",
								"content": nil,
								"tool_calls": []map[string]any{
									{
										"id":   "call_abc",
										"type": "function",
										"function": map[string]any{
											"name":      "get_weather",
											"arguments": `{"location":"London"}`,
										},
									},
								},
							},
						},
					},
					"usage": map[string]int{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
				})
			},
			wantToolCalls: 1,
		},
		{
			name: "no tool calls",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(chatResponse("I don't need tools for this."))
			},
			wantToolCalls: 0,
			wantContent:   "I don't need tools for this.",
		},
		{
			name: "server error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "fail", "type": "server_error"},
				})
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			defer srv.Close()

			client := newTestClient(srv.URL)
			msgs := []openai.ChatCompletionMessageParamUnion{
				{OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("What's the weather?"),
					},
				}},
			}
			tools := []openai.ChatCompletionToolParam{
				{
					Function: openai.FunctionDefinitionParam{
						Name:        "get_weather",
						Description: openai.String("Get weather for a location"),
						Parameters: openai.FunctionParameters{
							"type": "object",
							"properties": map[string]any{
								"location": map[string]any{"type": "string"},
							},
							"required": []string{"location"},
						},
					},
				},
			}

			resp, err := client.ChatWithTools(context.Background(), msgs, tools)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(resp.Choices) == 0 {
				t.Fatal("no choices returned")
			}

			got := len(resp.Choices[0].Message.ToolCalls)
			if got != tt.wantToolCalls {
				t.Errorf("tool_calls count = %d, want %d", got, tt.wantToolCalls)
			}
			if tt.wantContent != "" && resp.Choices[0].Message.Content != tt.wantContent {
				t.Errorf("content = %q, want %q", resp.Choices[0].Message.Content, tt.wantContent)
			}
		})
	}
}

func TestChatWithTools_RequestBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		tools, ok := req["tools"].([]any)
		if !ok || len(tools) == 0 {
			t.Error("tools not sent in request")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatResponse("no tools needed"))
	}))
	defer srv.Close()

	client := newTestClient(srv.URL)
	tools := []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name: "test_tool",
				Parameters: openai.FunctionParameters{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}
	msgs := []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String("test"),
			},
		}},
	}

	_, err := client.ChatWithTools(context.Background(), msgs, tools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- float64sToFloat32s ---

func TestFloat64sToFloat32s(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []float32
	}{
		{
			name:  "normal values",
			input: []float64{1.0, 2.5, -3.0},
			want:  []float32{1.0, 2.5, -3.0},
		},
		{
			name:  "empty",
			input: []float64{},
			want:  []float32{},
		},
		{
			name:  "single",
			input: []float64{0.42},
			want:  []float32{0.42},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float64sToFloat32s(tt.input)
			if len(got) != len(tt.want) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("[%d] = %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}
