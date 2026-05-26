package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/llm"
)

func TestAgentChat_NonStream(t *testing.T) {
	h, _ := setupTestHandler(t)

	srv := fakeChatCompletionServer(t, "Hola desde el agente")
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: srv.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, nil, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	sessionID := h.sessionStore.Create()
	body, _ := json.Marshal(AgentChatRequest{Message: "hola"})
	req := httptest.NewRequest(http.MethodPost, "/v1/agent/chat", bytes.NewReader(body))
	req.Header.Set("X-Session-ID", sessionID)
	w := httptest.NewRecorder()

	h.AgentChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("X-Session-ID"); got != sessionID {
		t.Errorf("X-Session-ID = %q, want %q", got, sessionID)
	}
	var resp AgentChatResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Content != "Hola desde el agente" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.SessionID != sessionID {
		t.Errorf("session_id = %q, want %q", resp.SessionID, sessionID)
	}
}

func TestAgentChat_EmptyMessageRejected(t *testing.T) {
	h, _ := setupTestHandler(t)
	ag := agent.NewAgent(nil, nil, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	// agent present but message empty -> 400 before agent use
	h.agentHandle.Set(ag)

	body, _ := json.Marshal(AgentChatRequest{Message: ""})
	req := httptest.NewRequest(http.MethodPost, "/v1/agent/chat", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.AgentChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestAgentChat_Stream(t *testing.T) {
	h, _ := setupTestHandler(t)

	srv := fakeStreamingChatServer(t, []string{"Hola", " mundo"}, "stop")
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: srv.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, nil, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	sessionID := h.sessionStore.Create()
	body, _ := json.Marshal(AgentChatRequest{Message: "saludos", Stream: true})
	req := httptest.NewRequest(http.MethodPost, "/v1/agent/chat", bytes.NewReader(body))
	req.Header.Set("X-Session-ID", sessionID)
	w := httptest.NewRecorder()

	h.AgentChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q", ct)
	}
	out := w.Body.String()

	// token events carry {"text": "..."}
	if !strings.Contains(out, "event: token") {
		t.Errorf("missing token event:\n%s", out)
	}
	if !strings.Contains(out, `"text":"Hola"`) || !strings.Contains(out, `"text":" mundo"`) {
		t.Errorf("missing token text:\n%s", out)
	}
	// terminal aggregate
	if !strings.Contains(out, "event: done") {
		t.Errorf("missing done event:\n%s", out)
	}
	// proprietary stream must NOT emit OpenAI chunk objects or [DONE]
	if strings.Contains(out, "chat.completion.chunk") || strings.Contains(out, "[DONE]") {
		t.Errorf("proprietary stream leaked OpenAI framing:\n%s", out)
	}
}

func TestAgentChat_Stream_EmitsToolEvents(t *testing.T) {
	h, _ := setupTestHandler(t)

	var callIdx int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		callIdx++
		if callIdx == 1 {
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{}"}}]},"finish_reason":null}]}`+"\n\n")
			fl.Flush()
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fl.Flush()
		} else {
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":null}]}`+"\n\n")
			fl.Flush()
			fmt.Fprint(w, `data: {"id":"u","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			fl.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	chatClient := llm.NewChatClient(llm.ChatClientConfig{
		BaseURL: srv.URL, Model: "test-model", MaxTokens: 100, Temperature: 0,
	})
	ag := agent.NewAgent(chatClient, []agent.Tool{stubEchoTool()}, "system", 5, nil, 0, 0, h.sessionStore, zerolog.Nop())
	h.agentHandle.Set(ag)

	body, _ := json.Marshal(AgentChatRequest{Message: "call echo", Stream: true})
	req := httptest.NewRequest(http.MethodPost, "/v1/agent/chat", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.AgentChat(w, req)

	out := w.Body.String()
	if !strings.Contains(out, "event: tool") {
		t.Errorf("missing tool event:\n%s", out)
	}
	if !strings.Contains(out, `"phase":"start"`) || !strings.Contains(out, `"phase":"done"`) {
		t.Errorf("missing tool phases:\n%s", out)
	}
	if !strings.Contains(out, `"name":"echo"`) {
		t.Errorf("missing tool name:\n%s", out)
	}
}
