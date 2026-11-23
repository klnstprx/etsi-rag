package agent

import (
	"context"
)

// StreamSink consumes the three streams produced by Agent.ChatStream and
// re-emits them in whatever wire format the caller needs. The OpenAI-shaped
// public endpoint and the admin HTMX/SSE handler both implement it.
//
// Any returned error aborts the relay immediately and is propagated to
// RelayChat's caller; this is how SSE handlers signal "client disconnected,
// stop writing."
type StreamSink interface {
	OnToken(token string) error
	OnToolEvent(evt ToolEvent) error
	// OnEnd is invoked exactly once after the agent's terminal outcome arrives
	// (or after the agent goroutine has otherwise drained). Exactly one of
	// result / agentErr is non-nil on a normal completion; both may be nil only
	// if no outcome was ever delivered (which shouldn't happen in practice).
	OnEnd(result *ChatResult, agentErr error) error
}

// RelayChat starts Agent.ChatStream and drains its tokens, tool events, and
// terminal outcome into the given sink. Returns whichever fires first: a sink
// error, a ctx error, or nil after OnEnd is called.
//
// Channel ownership: RelayChat creates the ToolEvent channel itself (the agent
// requires one but does not close it) and closes it after the agent's outcome
// arrives — at which point the agent goroutine has finished writing.
func RelayChat(ctx context.Context, ag *Agent, sessionID, userMessage string, sink StreamSink) error {
	eventChan := make(chan ToolEvent, 16)
	tokens, outcome := ag.ChatStream(ctx, sessionID, userMessage, eventChan)
	return drainStreams(ctx, tokens, eventChan, outcome, sink)
}

// drainStreams is the channel-orchestration core, extracted from RelayChat so
// tests can drive it with synthetic channels without spinning up a full agent.
//
// The caller passes the same eventChan they handed to ChatStream; drainStreams
// closes it once the outcome has been received (so any buffered tool events
// can still be drained before exit).
func drainStreams(
	ctx context.Context,
	tokens <-chan string,
	events chan ToolEvent,
	outcome <-chan ChatStreamOutcome,
	sink StreamSink,
) error {
	var (
		finalResult *ChatResult
		finalErr    error
		pending     <-chan ToolEvent = events // read side; nil once drained, while events stays the close handle
	)

	for {
		select {
		case <-ctx.Done():
			// We stop draining and return without closing events. The agent
			// goroutine is intentionally abandoned here: the same ctx is wired
			// into Agent.ChatStream, so runToolLoop unwinds on its own and the
			// goroutine exits. Its writes can't block beyond the tokens(64) /
			// events(16) buffers, so the abandonment window is bounded.
			return ctx.Err()

		case tok, ok := <-tokens:
			if !ok {
				tokens = nil
			} else if err := sink.OnToken(tok); err != nil {
				return err
			}

		case evt, ok := <-pending:
			if !ok {
				pending = nil
			} else if err := sink.OnToolEvent(evt); err != nil {
				return err
			}

		case out, ok := <-outcome:
			if ok {
				finalResult = out.Result
				finalErr = out.Err
			}
			outcome = nil
			// Agent has finished writing to events (runToolLoop ran to
			// completion before outcome was sent); safe to close so any
			// buffered tool events drain and then the select sees EOF.
			if events != nil {
				close(events)
				events = nil
			}
		}

		if tokens == nil && pending == nil && outcome == nil {
			return sink.OnEnd(finalResult, finalErr)
		}
	}
}
