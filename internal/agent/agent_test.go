package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"time"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/session"
	"etsi-rag/internal/testutil"
	"etsi-rag/internal/tokencount"
)

// --- Test helpers ---

// chatResp builds a chat completion JSON with no tool calls.
func chatResp(content string) map[string]any {
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
		"usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}
}

// toolCallResp builds a chat completion JSON with a single tool call.
func toolCallResp(toolName, toolCallID, argsJSON string) map[string]any {
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []map[string]any{
						{
							"id":   toolCallID,
							"type": "function",
							"function": map[string]any{
								"name":      toolName,
								"arguments": argsJSON,
							},
						},
					},
				},
			},
		},
		"usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}
}

// newSequentialLLM creates a test server that returns responses in order.
// Each call to /chat/completions returns the next response from the slice.
func newSequentialLLM(t *testing.T, responses []map[string]any) *httptest.Server {
	t.Helper()
	var callIdx atomic.Int64
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := int(callIdx.Add(1) - 1)
		if idx >= len(responses) {
			t.Errorf("LLM server received unexpected call #%d (only %d responses configured)", idx+1, len(responses))
			http.Error(w, "no more responses", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(responses[idx])
	}))
}

func newTestAgent(t *testing.T, srv *httptest.Server, tools []Tool, maxIter int) (*Agent, *session.Store) {
	t.Helper()
	return newTestAgentWithCounter(t, srv, tools, maxIter, nil, 0, 0)
}

func newTestAgentWithCounter(t *testing.T, srv *httptest.Server, tools []Tool, maxIter int, counter tokencount.Counter, contextWindow, toolOutputMaxTokens int) (*Agent, *session.Store) {
	t.Helper()
	client := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL:     srv.URL,
		Model:       "test-model",
		MaxTokens:   100,
		Temperature: 0,
	})
	ss := session.NewStore(config.SessionConfig{MaxMessages: 50, TTLSeconds: 3600})
	t.Cleanup(ss.Close)
	ag := NewAgent(client, tools, "You are a test assistant.", maxIter, counter, contextWindow, toolOutputMaxTokens, ss, zerolog.Nop())
	return ag, ss
}

// echoTool returns its arguments as the result.
func echoTool() Tool {
	return Tool{
		Name:        "echo",
		Description: "Echoes the input",
		Parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			return "echo: " + argsJSON, nil
		},
	}
}

// errorTool always returns an error.
func errorTool() Tool {
	return Tool{
		Name:        "fail",
		Description: "Always fails",
		Parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			return "", fmt.Errorf("tool failure")
		},
	}
}

// sourceTool appends a test source reference.
func sourceTool() Tool {
	return Tool{
		Name:        "search",
		Description: "Search with sources",
		Parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			if sources != nil {
				*sources = append(*sources, domain.SourceReference{
					ID:    "doc-1",
					Title: "Test Doc",
					Score: 0.95,
				})
			}
			return "found relevant content", nil
		},
	}
}

// --- Tests ---

func TestChat_SimpleResponse(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		chatResp("Hello!"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, nil, 5)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "Hi", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Hello!" {
		t.Errorf("response = %q, want %q", result.Response, "Hello!")
	}
	if len(result.Sources) != 0 {
		t.Errorf("expected no sources, got %d", len(result.Sources))
	}

	// Verify session has user + assistant messages
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	if len(msgs) != 2 {
		t.Fatalf("session has %d messages, want 2", len(msgs))
	}
	if msgs[0].OfUser == nil {
		t.Error("first message should be user")
	}
	if msgs[1].OfAssistant == nil {
		t.Error("second message should be assistant")
	}
}

func TestChat_WithToolCall(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("search", "call-1", `{"query":"test"}`),
		chatResp("Based on the search results..."),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{sourceTool()}, 5)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "Search for test", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Based on the search results..." {
		t.Errorf("response = %q, want %q", result.Response, "Based on the search results...")
	}

	// Verify sources were collected
	if len(result.Sources) != 1 {
		t.Fatalf("expected 1 source, got %d", len(result.Sources))
	}
	if result.Sources[0].ID != "doc-1" {
		t.Errorf("source ID = %q, want %q", result.Sources[0].ID, "doc-1")
	}

	// Session should have: user + assistant(tool_call) + tool_result + assistant(final) = 4
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	if len(msgs) != 4 {
		t.Errorf("session has %d messages, want 4", len(msgs))
	}
}

func TestChat_MaxIterationsExceeded(t *testing.T) {
	// LLM always returns a tool call — should hit max iterations
	responses := make([]map[string]any, 3)
	for i := range responses {
		responses[i] = toolCallResp("echo", fmt.Sprintf("call-%d", i), `{}`)
	}

	srv := newSequentialLLM(t, responses)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{echoTool()}, 3)
	sessionID := ss.Create()

	_, err := ag.Chat(context.Background(), sessionID, "loop forever", nil)
	if err == nil {
		t.Fatal("expected error for max iterations exceeded")
	}
	if got := err.Error(); got != "agent exceeded max tool iterations (3)" {
		t.Errorf("error = %q, want max iterations error", got)
	}
}

func TestChat_ToolError(t *testing.T) {
	// First call: tool call to "fail" tool, second call: final response
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("fail", "call-1", `{}`),
		chatResp("I encountered an error but recovered"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{errorTool()}, 5)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "try the broken tool", nil)
	if err != nil {
		t.Fatalf("Chat should not fail on tool error: %v", err)
	}
	if result.Response != "I encountered an error but recovered" {
		t.Errorf("response = %q, want recovery message", result.Response)
	}
}

func TestChat_SessionNotFound(t *testing.T) {
	srv := newSequentialLLM(t, nil) // no calls expected
	defer srv.Close()

	ag, _ := newTestAgent(t, srv, nil, 5)

	_, err := ag.Chat(context.Background(), "nonexistent-session", "hello", nil)
	if err == nil {
		t.Fatal("expected error for nonexistent session")
	}
}

func TestChatPassthrough(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		chatResp("Passthrough response"),
	})
	defer srv.Close()

	ag, _ := newTestAgent(t, srv, nil, 5)

	messages := []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String("Hello"),
			},
		}},
	}

	result, err := ag.ChatPassthrough(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("ChatPassthrough: %v", err)
	}
	if result.Response != "Passthrough response" {
		t.Errorf("response = %q, want %q", result.Response, "Passthrough response")
	}
}

// largeTool returns a tool that produces a result with the given number of words.
func largeTool(wordCount int) Tool {
	return Tool{
		Name:        "large",
		Description: "Returns a large result",
		Parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			words := make([]string, wordCount)
			for i := range words {
				words[i] = "word"
			}
			return strings.Join(words, " "), nil
		},
	}
}

func TestChat_ToolResultTooLarge(t *testing.T) {
	// contextWindow=1000, budget=250 tokens. Tool returns 300 words (= 300 tokens with MockCounter).
	// Agent should replace the result with an error message so the LLM can self-correct.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("I see the result was too large"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(300)}, 5, counter, 1000, 0)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "get big data", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "I see the result was too large" {
		t.Errorf("response = %q, want self-correction response", result.Response)
	}

	// Verify the tool result in session is the error feedback, not the 300-word blob.
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	for _, msg := range msgs {
		if msg.OfTool != nil {
			content := msg.OfTool.Content.OfString.Value
			if len(content) > 500 {
				t.Errorf("tool result in session should be error feedback, got %d chars", len(content))
			}
			if !strings.Contains(content, "too large") {
				t.Errorf("tool result should contain 'too large' feedback, got %q", content)
			}
		}
	}
}

// multiToolCallResp builds a chat completion JSON with multiple tool calls.
func multiToolCallResp(calls []struct{ name, id, args string }) map[string]any {
	toolCalls := make([]map[string]any, len(calls))
	for i, c := range calls {
		toolCalls[i] = map[string]any{
			"id":   c.id,
			"type": "function",
			"function": map[string]any{
				"name":      c.name,
				"arguments": c.args,
			},
		}
	}
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":       "assistant",
					"content":    "",
					"tool_calls": toolCalls,
				},
			},
		},
		"usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}
}

// collectEvents drains a ToolEvent channel into a slice.
func collectEvents(ch <-chan ToolEvent) []ToolEvent {
	var events []ToolEvent
	for evt := range ch {
		events = append(events, evt)
	}
	return events
}

func TestChat_ToolOutputMaxTokensOverride(t *testing.T) {
	// Explicit toolOutputMaxTokens=50 wins over contextWindow/4=250.
	// Tool returns 100 words → should be rejected by the tighter 50-token limit.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("Result was rejected"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(100)}, 5, counter, 1000, 50)
	sessionID := ss.Create()

	_, err := ag.Chat(context.Background(), sessionID, "get data", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	// Verify the tool result in session is the error feedback, not the 100-word blob.
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	var sawToolMsg bool
	for _, msg := range msgs {
		if msg.OfTool == nil {
			continue
		}
		sawToolMsg = true
		content := msg.OfTool.Content.OfString.Value
		if !strings.Contains(content, "too large") {
			t.Errorf("tool result should be rejected by explicit 50-token limit, got %q", content)
		}
		if !strings.Contains(content, "budget 50") {
			t.Errorf("error message should reference the configured budget of 50, got %q", content)
		}
	}
	if !sawToolMsg {
		t.Fatal("expected a tool message in session")
	}
}

func TestChat_ToolResultWithinBudget(t *testing.T) {
	// contextWindow=1000, budget=250 tokens. Tool returns 100 words — within budget.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("Got the data"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(100)}, 5, counter, 1000, 0)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "get data", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Got the data" {
		t.Errorf("response = %q, want %q", result.Response, "Got the data")
	}

	// Verify the tool result in session is the actual content, not truncated.
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	for _, msg := range msgs {
		if msg.OfTool != nil {
			content := msg.OfTool.Content.OfString.Value
			if strings.Contains(content, "too large") {
				t.Error("tool result should NOT be rejected — it's within budget")
			}
		}
	}
}

// --- Tool Event Tests ---

func TestChat_ToolEvents_Emitted(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("echo", "call-1", `{"msg":"hello"}`),
		chatResp("Done"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{echoTool()}, 5)
	sessionID := ss.Create()

	events := make(chan ToolEvent, 16)
	result, err := ag.Chat(context.Background(), sessionID, "test", events)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Done" {
		t.Errorf("response = %q, want %q", result.Response, "Done")
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 2 {
		t.Fatalf("got %d events, want 2", len(evts))
	}

	// Verify start event
	start := evts[0]
	if start.Type != "start" {
		t.Errorf("event[0].Type = %q, want %q", start.Type, "start")
	}
	if start.CallID != "call-1" {
		t.Errorf("event[0].CallID = %q, want %q", start.CallID, "call-1")
	}
	if start.Name != "echo" {
		t.Errorf("event[0].Name = %q, want %q", start.Name, "echo")
	}
	if start.Args != `{"msg":"hello"}` {
		t.Errorf("event[0].Args = %q, want %q", start.Args, `{"msg":"hello"}`)
	}

	// Verify done event
	done := evts[1]
	if done.Type != "done" {
		t.Errorf("event[1].Type = %q, want %q", done.Type, "done")
	}
	if done.CallID != "call-1" {
		t.Errorf("event[1].CallID = %q, want %q", done.CallID, "call-1")
	}
	if done.Name != "echo" {
		t.Errorf("event[1].Name = %q, want %q", done.Name, "echo")
	}
	if done.Duration <= 0 {
		t.Error("event[1].Duration should be > 0")
	}
	if done.Error {
		t.Error("event[1].Error should be false")
	}
	if !strings.Contains(done.Result, "echo:") {
		t.Errorf("event[1].Result = %q, should contain 'echo:'", done.Result)
	}
}

func TestChat_ToolEvents_NilChannel(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("echo", "call-1", `{}`),
		chatResp("Done"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{echoTool()}, 5)
	sessionID := ss.Create()

	// nil events channel should not panic
	result, err := ag.Chat(context.Background(), sessionID, "test", nil)
	if err != nil {
		t.Fatalf("Chat with nil events: %v", err)
	}
	if result.Response != "Done" {
		t.Errorf("response = %q, want %q", result.Response, "Done")
	}
}

func TestChat_ToolEvents_MultipleTools(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		multiToolCallResp([]struct{ name, id, args string }{
			{"echo", "call-1", `{"a":"1"}`},
			{"echo", "call-2", `{"b":"2"}`},
		}),
		chatResp("Both done"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{echoTool()}, 5)
	sessionID := ss.Create()

	events := make(chan ToolEvent, 16)
	result, err := ag.Chat(context.Background(), sessionID, "test", events)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Both done" {
		t.Errorf("response = %q, want %q", result.Response, "Both done")
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 4 {
		t.Fatalf("got %d events, want 4 (start/done for each tool)", len(evts))
	}

	// Verify order: start-1, done-1, start-2, done-2
	expected := []struct {
		typ    string
		callID string
	}{
		{"start", "call-1"},
		{"done", "call-1"},
		{"start", "call-2"},
		{"done", "call-2"},
	}
	for i, want := range expected {
		if evts[i].Type != want.typ || evts[i].CallID != want.callID {
			t.Errorf("event[%d] = {%s, %s}, want {%s, %s}",
				i, evts[i].Type, evts[i].CallID, want.typ, want.callID)
		}
	}
}

func TestChat_ToolEvents_Error(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("fail", "call-1", `{}`),
		chatResp("Recovered"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{errorTool()}, 5)
	sessionID := ss.Create()

	events := make(chan ToolEvent, 16)
	_, err := ag.Chat(context.Background(), sessionID, "test", events)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 2 {
		t.Fatalf("got %d events, want 2", len(evts))
	}

	done := evts[1]
	if !done.Error {
		t.Error("done event should have Error=true for failed tool")
	}
	if !strings.Contains(done.Result, "Error:") {
		t.Errorf("done.Result = %q, should contain 'Error:'", done.Result)
	}
}

func TestChat_ToolEvents_NoToolCalls(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		chatResp("Direct response"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, nil, 5)
	sessionID := ss.Create()

	events := make(chan ToolEvent, 16)
	result, err := ag.Chat(context.Background(), sessionID, "test", events)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "Direct response" {
		t.Errorf("response = %q, want %q", result.Response, "Direct response")
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 0 {
		t.Errorf("got %d events, want 0 (no tool calls)", len(evts))
	}
}

func TestChatPassthrough_ToolEvents(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("echo", "call-1", `{}`),
		chatResp("Passthrough done"),
	})
	defer srv.Close()

	ag, _ := newTestAgent(t, srv, []Tool{echoTool()}, 5)

	messages := []openai.ChatCompletionMessageParamUnion{
		{OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String("test"),
			},
		}},
	}

	events := make(chan ToolEvent, 16)
	result, err := ag.ChatPassthrough(context.Background(), messages, events)
	if err != nil {
		t.Fatalf("ChatPassthrough: %v", err)
	}
	if result.Response != "Passthrough done" {
		t.Errorf("response = %q, want %q", result.Response, "Passthrough done")
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 2 {
		t.Fatalf("got %d events, want 2", len(evts))
	}
	if evts[0].Type != "start" || evts[1].Type != "done" {
		t.Error("expected start+done events")
	}
}

func TestChat_ToolEvents_Duration(t *testing.T) {
	// Use a tool that sleeps briefly to verify duration tracking.
	sleepTool := Tool{
		Name:        "slow",
		Description: "Sleeps briefly",
		Parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			time.Sleep(50 * time.Millisecond)
			return "done", nil
		},
	}

	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("slow", "call-1", `{}`),
		chatResp("OK"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{sleepTool}, 5)
	sessionID := ss.Create()

	events := make(chan ToolEvent, 16)
	_, err := ag.Chat(context.Background(), sessionID, "test", events)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	close(events)
	evts := collectEvents(events)
	if len(evts) != 2 {
		t.Fatalf("got %d events, want 2", len(evts))
	}

	done := evts[1]
	if done.Duration < 50*time.Millisecond {
		t.Errorf("duration = %v, want >= 50ms", done.Duration)
	}
	if done.Duration > 5*time.Second {
		t.Errorf("duration = %v, unreasonably long", done.Duration)
	}
}
