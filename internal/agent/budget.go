package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"etsi-rag/internal/tokencount"
)

// schemaFramingMargin pads the measured fixed cost. Chat templates render the
// system prompt and tool schemas with extra framing (role markers, JSON
// punctuation, tool-call delimiters) that the raw token count of the prompt
// text and the bare schema JSON does not capture. 10% covers the observed gap.
const schemaFramingMargin = 1.10

// Budget is the per-request token budget the agent derives its guards from.
//
// All values are tokens. ToolPool is what's left for a single tool result
// after carving out generation headroom, the fixed prompt/schema cost, and the
// reserves for the incoming user message and session history.
type Budget struct {
	ContextWindow  int // effective per-request window (probed slot n_ctx)
	Generation     int // reserved for model output (agent.max_tokens)
	Fixed          int // measured system prompt + tool schema cost
	UserReserve    int // headroom for the incoming user message
	HistoryReserve int // headroom for session history
}

// ToolPool returns the token budget available to a single tool result:
// ContextWindow minus generation, fixed, and the two reserves, clamped to >= 0.
func (b Budget) ToolPool() int {
	pool := b.ContextWindow - b.Generation - b.Fixed - b.UserReserve - b.HistoryReserve
	if pool < 0 {
		return 0
	}
	return pool
}

// MeasureFixed measures the fixed per-request token cost: the system prompt
// plus every tool's definition (the same openai.ChatCompletionToolParam
// produced by toFunctionTool, marshalled to JSON), summed and padded by
// schemaFramingMargin. The result is rounded up.
//
// If counter.Count errors, the error is returned unchanged — the CALLER
// decides the fallback (e.g. a chars-per-token estimate), because only the
// caller knows whether a conservative over-estimate or hard failure is right
// for its context.
func MeasureFixed(ctx context.Context, counter tokencount.Counter, system string, tools []Tool) (int, error) {
	total, err := counter.Count(ctx, system)
	if err != nil {
		return 0, fmt.Errorf("counting system prompt: %w", err)
	}

	for _, t := range tools {
		schema, err := json.Marshal(toFunctionTool(t))
		if err != nil {
			return 0, fmt.Errorf("marshalling tool schema for %q: %w", t.Name(), err)
		}
		n, err := counter.Count(ctx, string(schema))
		if err != nil {
			return 0, fmt.Errorf("counting tool schema for %q: %w", t.Name(), err)
		}
		total += n
	}

	return int(math.Ceil(float64(total) * schemaFramingMargin)), nil
}
