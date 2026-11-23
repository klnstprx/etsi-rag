package agent

import (
	"context"
	"encoding/json"
	"errors"
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
	return newTestAgentWithCounter(t, srv, tools, maxIter, nil, Budget{}, 0)
}

func newTestAgentWithCounter(t *testing.T, srv *httptest.Server, tools []Tool, maxIter int, counter tokencount.Counter, budget Budget, toolOutputMaxTokens int) (*Agent, *session.Store) {
	t.Helper()
	client := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL:     srv.URL,
		Model:       "test-model",
		MaxTokens:   100,
		Temperature: 0,
	})
	ss := session.NewStore(config.SessionConfig{MaxMessages: 50, TTLSeconds: 3600})
	t.Cleanup(ss.Close)
	ag := NewAgent(client, tools, "You are a test assistant.", maxIter, counter, budget, toolOutputMaxTokens, ss, zerolog.Nop())
	return ag, ss
}

// budgetWithPool returns a Budget whose ToolPool() equals exactly pool tokens
// (all headroom folded into Generation), so checkToolResult tests can pin the
// derived cap without restating the subtraction.
func budgetWithPool(window, pool int) Budget {
	return Budget{ContextWindow: window, Generation: window - pool}
}

// stubTool is a minimal Tool implementation for tests.
type stubTool struct {
	name        string
	description string
	parameters  openai.FunctionParameters
	execute     func(ctx context.Context, argsJSON string) (ToolResult, error)
}

func (t *stubTool) Name() string                          { return t.name }
func (t *stubTool) Description() string                   { return t.description }
func (t *stubTool) Parameters() openai.FunctionParameters { return t.parameters }
func (t *stubTool) Execute(ctx context.Context, argsJSON string) (ToolResult, error) {
	return t.execute(ctx, argsJSON)
}

func newStubTool(name, description string, exec func(ctx context.Context, argsJSON string) (ToolResult, error)) Tool {
	return &stubTool{
		name:        name,
		description: description,
		parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		execute:     exec,
	}
}

// echoTool returns its arguments as the result.
func echoTool() Tool {
	return newStubTool("echo", "Echoes the input", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		return ToolResult{Text: "echo: " + argsJSON}, nil
	})
}

// imagesTool returns a tool result with a citation, no images.
func imagesToolWithRefs(images []ImageRef) Tool {
	return newStubTool("search_with_images", "Search returning images", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		return ToolResult{
			Text:   "found relevant content (image markers stripped)",
			Images: images,
		}, nil
	})
}

// errorTool always returns an error.
func errorTool() Tool {
	return newStubTool("fail", "Always fails", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		return ToolResult{}, fmt.Errorf("tool failure")
	})
}

// sourceTool returns a citation along with its result.
func sourceTool() Tool {
	return newStubTool("search", "Search with sources", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		return ToolResult{
			Text:      "found relevant content",
			Citations: []Citation{{Title: "Test Doc", URL: "https://example.com/doc-1"}},
		}, nil
	})
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
	if len(result.Citations) != 0 {
		t.Errorf("expected no citations, got %d", len(result.Citations))
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

	// Verify citations were collected
	if len(result.Citations) != 1 {
		t.Fatalf("expected 1 citation, got %d", len(result.Citations))
	}
	if result.Citations[0].Title != "Test Doc" {
		t.Errorf("citation title = %q, want %q", result.Citations[0].Title, "Test Doc")
	}
	if result.Citations[0].URL != "https://example.com/doc-1" {
		t.Errorf("citation URL = %q, want %q", result.Citations[0].URL, "https://example.com/doc-1")
	}

	// Tool calls must be recorded for downstream observability (eval harness,
	// debugging which tools fired without parsing logs).
	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call record, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "search" {
		t.Errorf("ToolCalls[0].Name = %q, want %q", result.ToolCalls[0].Name, "search")
	}
	if result.ToolCalls[0].Args == "" {
		t.Errorf("ToolCalls[0].Args is empty; expected the LLM-supplied JSON args")
	}
	if result.ToolCalls[0].Error {
		t.Errorf("ToolCalls[0].Error should be false on a successful call")
	}

	// Session should have: user + assistant(tool_call) + tool_result + assistant(final) = 4
	sess := ss.Get(sessionID)
	msgs := sess.GetMessages()
	if len(msgs) != 4 {
		t.Errorf("session has %d messages, want 4", len(msgs))
	}
}

// Two tool calls returning the same citations and images must yield one
// deduplicated entry each in the final ChatResult — the UI renders these
// lists directly.
func TestChat_DeduplicatesCitationsAndImagesAcrossToolCalls(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("search", "call-1", `{"query":"a"}`),
		toolCallResp("search", "call-2", `{"query":"b"}`),
		chatResp("done"),
	})
	defer srv.Close()

	tool := newStubTool("search", "Search", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		return ToolResult{
			Text: "content",
			Citations: []Citation{
				{Title: "Doc", URL: "https://example.com/doc-1"},
				{Title: "Untitled"}, // no URL: dedupe by title
			},
			Images: []ImageRef{{DocID: "d1", Ref: "p1", URL: "/v1/artifacts/tok.png"}},
		}, nil
	})

	ag, ss := newTestAgent(t, srv, []Tool{tool}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "http://localhost:8080" })
	sessionID := ss.Create()

	res, err := ag.Chat(context.Background(), sessionID, "q", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(res.Citations) != 2 {
		t.Errorf("Citations len = %d, want 2 (deduplicated across tool calls), got %+v", len(res.Citations), res.Citations)
	}
	if len(res.Images) != 1 {
		t.Errorf("Images len = %d, want 1 (deduplicated across tool calls), got %+v", len(res.Images), res.Images)
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

// largeTool returns a tool that produces a result with the given number of words.
func largeTool(wordCount int) Tool {
	return newStubTool("large", "Returns a large result", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		words := make([]string, wordCount)
		for i := range words {
			words[i] = "word"
		}
		return ToolResult{Text: strings.Join(words, " ")}, nil
	})
}

func TestChat_ToolResultTooLarge(t *testing.T) {
	// ToolPool()=400 tokens (NOT contextWindow/4=250 — proves the cap derives
	// from the budget pool, not the old /4 rule). Tool returns 500 words
	// (= 500 tokens with MockCounter), so it exceeds the 400-token pool and the
	// agent replaces the result with an error message so the LLM can self-correct.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("I see the result was too large"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(500)}, 5, counter, budgetWithPool(1000, 400), 0)
	sessionID := ss.Create()

	result, err := ag.Chat(context.Background(), sessionID, "get big data", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if result.Response != "I see the result was too large" {
		t.Errorf("response = %q, want self-correction response", result.Response)
	}

	// Verify the tool result in session is the error feedback, not the 500-word blob.
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
			// The cap must equal ToolPool()=400, not the old contextWindow/4=250.
			if !strings.Contains(content, "budget 400") {
				t.Errorf("error message should reference the ToolPool budget of 400, got %q", content)
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
	// Explicit toolOutputMaxTokens=50 wins over the derived ToolPool()=400.
	// Tool returns 100 words → should be rejected by the tighter 50-token limit.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("Result was rejected"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(100)}, 5, counter, budgetWithPool(1000, 400), 50)
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
	// ToolPool()=400 tokens. Tool returns 100 words — within budget.
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("Got the data"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(100)}, 5, counter, budgetWithPool(1000, 400), 0)
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

// TestChat_ToolResultPoolFloor asserts the 256-token floor: even when the
// derived ToolPool() is tiny (or zero), the cap never drops below 256, so a
// modest tool result isn't rejected by an over-subscribed budget.
func TestChat_ToolResultPoolFloor(t *testing.T) {
	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("large", "call-1", `{}`),
		chatResp("ok"),
	})
	defer srv.Close()

	counter := &testutil.MockCounter{}
	// ToolPool()=0 (fully over-subscribed) → cap floored at 256. A 100-word
	// result must pass.
	ag, ss := newTestAgentWithCounter(t, srv, []Tool{largeTool(100)}, 5, counter, budgetWithPool(1000, 0), 0)
	sessionID := ss.Create()

	if _, err := ag.Chat(context.Background(), sessionID, "get data", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	sess := ss.Get(sessionID)
	for _, msg := range sess.GetMessages() {
		if msg.OfTool != nil {
			if strings.Contains(msg.OfTool.Content.OfString.Value, "too large") {
				t.Error("100-word result should pass under the 256-token floor")
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
	if done.Args != `{"msg":"hello"}` {
		t.Errorf("event[1].Args = %q, want %q", done.Args, `{"msg":"hello"}`)
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

func TestChat_ToolEvents_Duration(t *testing.T) {
	// Use a tool that sleeps briefly to verify duration tracking.
	sleepTool := newStubTool("slow", "Sleeps briefly", func(ctx context.Context, argsJSON string) (ToolResult, error) {
		time.Sleep(50 * time.Millisecond)
		return ToolResult{Text: "done"}, nil
	})

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

// recordingLLM is like newSequentialLLM but also records every request body
// so tests can introspect what the agent sent to the model.
func recordingLLM(t *testing.T, responses []map[string]any, recorded *[]map[string]any) *httptest.Server {
	t.Helper()
	var callIdx atomic.Int64
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding LLM request body: %v", err)
		}
		*recorded = append(*recorded, body)
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

// findImageURLs walks a recorded request's messages and returns every
// image_url URL found across all user-message content parts. Used to assert
// the agent injected the synthetic user message with image parts.
func findImageURLs(req map[string]any) []string {
	var out []string
	msgs, _ := req["messages"].([]any)
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] != "user" {
			continue
		}
		parts, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			pp, _ := p.(map[string]any)
			if pp["type"] != "image_url" {
				continue
			}
			iu, _ := pp["image_url"].(map[string]any)
			if u, ok := iu["url"].(string); ok {
				out = append(out, u)
			}
		}
	}
	return out
}

// TestChat_ImagesAttachedWhenVisionEnabled asserts that when the agent's
// VisionGetter reports vision=true with a baseURL and the tool returns
// Images, the agent appends a synthetic user message containing image_url
// content parts in the SECOND request to the LLM (the one after the tool
// message).
func TestChat_ImagesAttachedWhenVisionEnabled(t *testing.T) {
	images := []ImageRef{
		{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"},
		{DocID: "doc-1", Ref: "p2", URL: "/v1/artifacts/bbb.png"},
	}

	var recorded []map[string]any
	srv := recordingLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("here is the answer based on the images"),
	}, &recorded)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "http://localhost:8080" })

	sessID := ss.Create()

	res, err := ag.Chat(context.Background(), sessID, "describe pages", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if res.Response == "" {
		t.Fatal("empty response")
	}
	if len(res.Images) != 2 {
		t.Fatalf("ChatResult.Images len = %d, want 2", len(res.Images))
	}

	if len(recorded) < 2 {
		t.Fatalf("expected 2 LLM requests, got %d", len(recorded))
	}

	urls := findImageURLs(recorded[1])
	if len(urls) != 2 {
		t.Fatalf("expected 2 image_url parts in second request, got %d", len(urls))
	}
	wantPrefixes := []string{
		"http://localhost:8080/v1/artifacts/aaa.png",
		"http://localhost:8080/v1/artifacts/bbb.png",
	}
	for i, want := range wantPrefixes {
		if urls[i] != want {
			t.Errorf("image url[%d] = %q, want %q", i, urls[i], want)
		}
	}
}

// TestChat_ImagesNotAttachedWhenVisionDisabled asserts the inverse: same
// tool returns Images, but VisionGetter says false → no image_url parts in
// any request.
func TestChat_ImagesNotAttachedWhenVisionDisabled(t *testing.T) {
	images := []ImageRef{{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"}}

	var recorded []map[string]any
	srv := recordingLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("text-only answer"),
	}, &recorded)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	// Default: no vision getter set → SupportsVision treated as false.

	sessID := ss.Create()

	res, err := ag.Chat(context.Background(), sessID, "describe pages", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(res.Images) != 0 {
		t.Errorf("ChatResult.Images should be empty when vision disabled, got %d", len(res.Images))
	}
	for i, req := range recorded {
		if urls := findImageURLs(req); len(urls) > 0 {
			t.Errorf("request[%d] unexpectedly has image_url parts: %v", i, urls)
		}
	}
}

// TestChat_ToolEventCarriesImagesEvenWhenVisionOff is the observability
// invariant: ToolEvent.Images must always reflect what the tool resolved
// (so the UI can show thumbnails / links) regardless of whether vision was
// enabled. VisionAttached distinguishes "actually sent to LLM" from "just
// available for reference". Without this, a misconfigured backend gives no
// UI signal that images existed.
func TestChat_ToolEventCarriesImagesEvenWhenVisionOff(t *testing.T) {
	images := []ImageRef{
		{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"},
		{DocID: "doc-1", Ref: "p2", URL: "/v1/artifacts/bbb.png"},
	}

	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("text-only answer"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	// No vision getter → vision off, but tool still resolved images.

	events := make(chan ToolEvent, 8)
	sessID := ss.Create()
	if _, err := ag.Chat(context.Background(), sessID, "x", events); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	close(events)

	var doneEvt *ToolEvent
	for evt := range events {
		if evt.Type == "done" {
			doneEvt = &evt
		}
	}
	if doneEvt == nil {
		t.Fatal("no done event observed")
	}
	if len(doneEvt.Images) != 2 {
		t.Errorf("ToolEvent.Images len = %d, want 2 (observability: tool's resolved images surface regardless of vision config)", len(doneEvt.Images))
	}
	if doneEvt.VisionAttached {
		t.Error("VisionAttached should be false when vision is off")
	}
}

// TestChat_ToolEventVisionAttachedWhenEnabled is the positive complement:
// when vision IS enabled, VisionAttached=true and Images carries the same
// list (the SDK shape was already verified by other tests).
func TestChat_ToolEventVisionAttachedWhenEnabled(t *testing.T) {
	images := []ImageRef{{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"}}

	srv := newSequentialLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("answer with image context"),
	})
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "http://localhost:8080" })

	events := make(chan ToolEvent, 8)
	sessID := ss.Create()
	if _, err := ag.Chat(context.Background(), sessID, "x", events); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	close(events)

	var doneEvt *ToolEvent
	for evt := range events {
		if evt.Type == "done" {
			doneEvt = &evt
		}
	}
	if doneEvt == nil {
		t.Fatal("no done event observed")
	}
	if len(doneEvt.Images) != 1 {
		t.Errorf("ToolEvent.Images len = %d, want 1", len(doneEvt.Images))
	}
	if !doneEvt.VisionAttached {
		t.Error("VisionAttached should be true when vision is enabled")
	}
}

// TestChat_ImagesInlinedWhenBaseURLEmpty covers inline transport: with no
// base URL and an inliner installed, each image_url carries the inliner's
// data URI, so the backend never has to fetch anything.
func TestChat_ImagesInlinedWhenBaseURLEmpty(t *testing.T) {
	images := []ImageRef{
		{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"},
		{DocID: "doc-1", Ref: "p2", URL: "/v1/artifacts/bbb.png"},
	}

	var recorded []map[string]any
	srv := recordingLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("answer"),
	}, &recorded)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "" })
	ag.SetImageInliner(func(_ context.Context, img ImageRef) (string, error) {
		return "data:image/png;base64," + img.Ref, nil
	})

	res, err := ag.Chat(context.Background(), ss.Create(), "x", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(res.Images) != 2 {
		t.Fatalf("ChatResult.Images len = %d, want 2", len(res.Images))
	}
	urls := findImageURLs(recorded[1])
	want := []string{"data:image/png;base64,p1", "data:image/png;base64,p2"}
	if len(urls) != len(want) {
		t.Fatalf("image_url parts = %v, want %v", urls, want)
	}
	for i := range want {
		if urls[i] != want[i] {
			t.Errorf("image url[%d] = %q, want %q", i, urls[i], want[i])
		}
	}
}

// TestChat_InlineFailureDropsOnlyThatImage: an image that cannot be loaded
// is left out, and the rest are still attached and the turn still answers.
func TestChat_InlineFailureDropsOnlyThatImage(t *testing.T) {
	images := []ImageRef{
		{DocID: "doc-1", Ref: "missing", URL: "/v1/artifacts/aaa.png"},
		{DocID: "doc-1", Ref: "p2", URL: "/v1/artifacts/bbb.png"},
	}

	var recorded []map[string]any
	srv := recordingLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("answer"),
	}, &recorded)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "" })
	ag.SetImageInliner(func(_ context.Context, img ImageRef) (string, error) {
		if img.Ref == "missing" {
			return "", errors.New("object not found")
		}
		return "data:image/png;base64," + img.Ref, nil
	})

	res, err := ag.Chat(context.Background(), ss.Create(), "x", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if res.Response != "answer" {
		t.Errorf("Response = %q, want %q", res.Response, "answer")
	}
	if len(res.Images) != 1 || res.Images[0].Ref != "p2" {
		t.Errorf("ChatResult.Images = %v, want only p2", res.Images)
	}
	if urls := findImageURLs(recorded[1]); len(urls) != 1 || urls[0] != "data:image/png;base64,p2" {
		t.Errorf("image_url parts = %v, want only p2's data URI", urls)
	}
}

// TestChat_ImagesNotAttachedWhenBaseURLEmpty covers the partial-config case:
// SupportsVision=true but PublicBaseURL is empty. Expected behavior: no
// images attached, since the LLM backend would fail to fetch a path-only URL.
func TestChat_ImagesNotAttachedWhenBaseURLEmpty(t *testing.T) {
	images := []ImageRef{{DocID: "doc-1", Ref: "p1", URL: "/v1/artifacts/aaa.png"}}

	var recorded []map[string]any
	srv := recordingLLM(t, []map[string]any{
		toolCallResp("search_with_images", "tc1", `{"query":"x"}`),
		chatResp("answer"),
	}, &recorded)
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, []Tool{imagesToolWithRefs(images)}, 5)
	ag.SetVisionGetter(func() (bool, string) { return true, "" })

	sessID := ss.Create()

	if _, err := ag.Chat(context.Background(), sessID, "x", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	for i, req := range recorded {
		if urls := findImageURLs(req); len(urls) > 0 {
			t.Errorf("request[%d] unexpectedly has image_url parts: %v", i, urls)
		}
	}
}
