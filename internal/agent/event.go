package agent

import "time"

// ToolEvent represents a tool execution lifecycle event emitted during the
// agent's tool-calling loop. Consumers (e.g., SSE handlers) use these to
// provide real-time feedback on tool activity.
type ToolEvent struct {
	Type     string        // "start" or "done"
	CallID   string        // unique tool call ID from the LLM response
	Name     string        // tool name (e.g., "rag_search")
	Args     string        // JSON arguments (start only)
	Result   string        // tool result text (done only)
	Duration time.Duration // wall-clock execution time (done only)
	Error    bool          // true if tool returned an error (done only)
}
