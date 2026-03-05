// Package domain contains core domain types.
// Note: For chat messages, use github.com/openai/openai-go types directly.
package domain

// ConversationHistory manages a bounded conversation history.
// Used for session management to track conversation context.
type ConversationHistory struct {
	// MessageCount tracks the number of messages.
	MessageCount int

	// MaxLength is the maximum number of messages to retain.
	MaxLength int

	// Note: Actual messages are stored as openai.ChatCompletionMessageParamUnion
	// in the session store. This struct just tracks metadata.
}

// NewConversationHistory creates a new history tracker.
func NewConversationHistory(maxLength int) *ConversationHistory {
	return &ConversationHistory{
		MessageCount: 0,
		MaxLength:    maxLength,
	}
}
