package agent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// recordingSink records every callback the relay makes, in order, so tests can
// assert both the dispatch contract and ordering. Each method optionally returns
// an injected error to exercise sink-aborts.
type recordingSink struct {
	mu        sync.Mutex
	events    []string // "tok:<s>", "tool_start:<name>", "tool_done:<name>", "end:<result-or-err>"
	tokenErr  error
	toolErr   error
	endErr    error
	endResult *ChatResult
	endAgent  error
}

func (s *recordingSink) OnToken(tok string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, "tok:"+tok)
	return s.tokenErr
}

func (s *recordingSink) OnToolEvent(evt ToolEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, "tool_"+evt.Type+":"+evt.Name)
	return s.toolErr
}

func (s *recordingSink) OnEnd(result *ChatResult, agentErr error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.endResult = result
	s.endAgent = agentErr
	switch {
	case agentErr != nil:
		s.events = append(s.events, "end:err="+agentErr.Error())
	case result != nil:
		s.events = append(s.events, "end:result="+result.Response)
	default:
		s.events = append(s.events, "end:nil")
	}
	return s.endErr
}

func (s *recordingSink) snapshot() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.events))
	copy(out, s.events)
	return out
}

// drainScript drives drainStreams from a deterministic script that emits onto
// the three channels in the order the agent goroutine would, then closes
// tokens / outcome the way Agent.ChatStream's deferred closes do.
func drainScript(t *testing.T, ctx context.Context, sink StreamSink, script func(tokens chan<- string, events chan<- ToolEvent, outcome chan<- ChatStreamOutcome)) error {
	t.Helper()
	tokens := make(chan string, 8)
	events := make(chan ToolEvent, 8)
	outcome := make(chan ChatStreamOutcome, 1)

	go script(tokens, events, outcome)
	return drainStreams(ctx, tokens, events, outcome, sink)
}

func TestDrainStreams_TokensOnly(t *testing.T) {
	sink := &recordingSink{}
	err := drainScript(t, context.Background(), sink, func(tokens chan<- string, events chan<- ToolEvent, outcome chan<- ChatStreamOutcome) {
		tokens <- "Hello"
		tokens <- ", "
		tokens <- "world"
		outcome <- ChatStreamOutcome{Result: &ChatResult{Response: "Hello, world"}}
		close(outcome)
		close(tokens)
	})
	if err != nil {
		t.Fatalf("drainStreams err: %v", err)
	}
	got := sink.snapshot()
	want := []string{"tok:Hello", "tok:, ", "tok:world", "end:result=Hello, world"}
	if !equalSeq(got, want) {
		t.Fatalf("event sequence mismatch:\n got  %v\n want %v", got, want)
	}
}

func TestDrainStreams_InterleavedTokensAndToolEvents(t *testing.T) {
	sink := &recordingSink{}
	err := drainScript(t, context.Background(), sink, func(tokens chan<- string, events chan<- ToolEvent, outcome chan<- ChatStreamOutcome) {
		// Real loop interleaves: think tokens, then tool start/done, then more tokens.
		// We can't guarantee ordering across separate channels in the select, but each
		// individual channel is ordered. The test only asserts per-channel ordering.
		events <- ToolEvent{Type: "start", Name: "semantic_search"}
		events <- ToolEvent{Type: "done", Name: "semantic_search"}
		tokens <- "Answer"
		outcome <- ChatStreamOutcome{Result: &ChatResult{Response: "Answer"}}
		close(outcome)
		close(tokens)
	})
	if err != nil {
		t.Fatalf("drainStreams err: %v", err)
	}
	got := sink.snapshot()
	// Per-channel ordering only: tool_start before tool_done; everything before end.
	startIdx := idxOf(got, "tool_start:semantic_search")
	doneIdx := idxOf(got, "tool_done:semantic_search")
	tokIdx := idxOf(got, "tok:Answer")
	endIdx := idxOf(got, "end:result=Answer")
	if startIdx == -1 || doneIdx == -1 || tokIdx == -1 || endIdx == -1 {
		t.Fatalf("missing required event in sequence: %v", got)
	}
	if startIdx >= doneIdx {
		t.Errorf("tool_start must precede tool_done: %v", got)
	}
	if tokIdx >= endIdx || startIdx >= endIdx || doneIdx >= endIdx {
		t.Errorf("end must be last: %v", got)
	}
}

func TestDrainStreams_AgentError(t *testing.T) {
	sink := &recordingSink{}
	err := drainScript(t, context.Background(), sink, func(tokens chan<- string, events chan<- ToolEvent, outcome chan<- ChatStreamOutcome) {
		outcome <- ChatStreamOutcome{Err: errors.New("agent boom")}
		close(outcome)
		close(tokens)
	})
	if err != nil {
		t.Fatalf("drainStreams unexpectedly returned err: %v", err)
	}
	got := sink.snapshot()
	if len(got) != 1 || got[0] != "end:err=agent boom" {
		t.Fatalf("want single end:err entry, got %v", got)
	}
}

func TestDrainStreams_SinkErrorAbortsEarly(t *testing.T) {
	sink := &recordingSink{tokenErr: errors.New("sink dead")}
	tokens := make(chan string, 8)
	events := make(chan ToolEvent, 8)
	outcome := make(chan ChatStreamOutcome, 1)
	tokens <- "first"
	// don't close: drain should abort on the first sink.OnToken error
	err := drainStreams(context.Background(), tokens, events, outcome, sink)
	if err == nil || err.Error() != "sink dead" {
		t.Fatalf("want sink dead, got %v", err)
	}
	got := sink.snapshot()
	if len(got) != 1 || got[0] != "tok:first" {
		t.Fatalf("want only first token recorded, got %v", got)
	}
}

func TestDrainStreams_ContextCancelled(t *testing.T) {
	sink := &recordingSink{}
	tokens := make(chan string)
	events := make(chan ToolEvent)
	outcome := make(chan ChatStreamOutcome, 1)
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		done <- drainStreams(ctx, tokens, events, outcome, sink)
	}()

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("drainStreams did not return after ctx cancel")
	}
}

// TestRelayChat_EndToEnd exercises the full RelayChat against a real Agent
// backed by an SSE-emitting fake LLM, confirming that the wrapper correctly
// wires Agent.ChatStream channels into the sink.
func TestRelayChat_EndToEnd(t *testing.T) {
	srv := newStreamingLLM(t, []string{"Hola", " mundo"}, "stop")
	defer srv.Close()

	ag, ss := newTestAgent(t, srv, nil, 1)
	sid := ss.Create()

	sink := &recordingSink{}
	err := RelayChat(context.Background(), ag, sid, "saludos", sink)
	if err != nil {
		t.Fatalf("RelayChat err: %v", err)
	}

	got := sink.snapshot()
	if idxOf(got, "tok:Hola") == -1 || idxOf(got, "tok: mundo") == -1 {
		t.Fatalf("missing expected token events: %v", got)
	}
	if got[len(got)-1] != "end:result=Hola mundo" {
		t.Fatalf("end mismatch: %v", got)
	}
}

// newStreamingLLM returns an httptest server whose /chat/completions endpoint
// emits the given content tokens as OpenAI SSE chunks, followed by a finish
// chunk and [DONE].
func newStreamingLLM(t *testing.T, contents []string, finishReason string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, c := range contents {
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}]}\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", finishReason)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

// equalSeq is a small slice equality helper.
func equalSeq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// containsSeq returns true if `sub` appears as a contiguous subsequence of `s`.
func containsSeq(s, sub []string) bool {
	if len(sub) == 0 {
		return true
	}
outer:
	for i := 0; i+len(sub) <= len(s); i++ {
		for j := range sub {
			if s[i+j] != sub[j] {
				continue outer
			}
		}
		return true
	}
	return false
}

// idxOf returns the index of v in s, or -1.
func idxOf(s []string, v string) int {
	for i, e := range s {
		if e == v {
			return i
		}
	}
	return -1
}
