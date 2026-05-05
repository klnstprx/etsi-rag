package agent

import "time"

// ToolEvent represents a tool execution lifecycle event emitted during the
// agent's tool-calling loop. Consumers (e.g., SSE handlers) use these to
// provide real-time feedback on tool activity.
type ToolEvent struct {
	Type      string        // "start" or "done"
	CallID    string        // unique tool call ID from the LLM response
	Name      string        // tool name (e.g., "semantic_search")
	Args      string        // JSON arguments (start and done)
	Result    string        // tool result text (done only)
	Duration  time.Duration // wall-clock execution time (done only)
	Error     bool          // true if tool returned an error (done only)
	Citations []Citation    // external references produced by this call (done only)

	// Images lists every image artifact the tool resolved from chunk markers,
	// regardless of whether they were forwarded to the LLM. This is for
	// UI/observability — the user always sees what visual context exists for
	// the cited documents. Empty when the tool found no markers.
	Images []ImageRef

	// VisionAttached is true when the agent actually injected the images as
	// image_url content parts in a synthetic user message (i.e. the
	// configured backend supports vision and PublicBaseURL is set). When
	// false with non-empty Images, the UI should show a "not attached"
	// indicator so the misconfiguration is visible.
	VisionAttached bool
}
