package agent

import (
	"context"

	"github.com/openai/openai-go"
)

// Tool is a callable tool for the agent's tool-calling loop. Each concrete
// tool (semantic_search, web_search, weather, …) is its own type implementing
// this interface.
type Tool interface {
	Name() string
	Description() string
	Parameters() openai.FunctionParameters
	Execute(ctx context.Context, argsJSON string) (ToolResult, error)
}

// ToolResult is the output of a tool call.
// Text is what the LLM sees on the tool message. Citations are optional
// UI-facing references to external resources the tool used; tools without
// external references (weather, date, calculator) leave this nil. Images
// are optional vision artifacts the agent will attach to a synthetic
// follow-up user message when the configured backend supports vision.
type ToolResult struct {
	Text      string
	Citations []Citation
	Images    []ImageRef
}

// Citation is a tool-agnostic reference to an external resource (document,
// webpage, PDF). Both semantic search and web search produce these.
type Citation struct {
	Title string
	URL   string // may be empty when no URL is known
}

// ImageRef points to an image artifact derivable from a source document. The
// agent uses URL as the image_url for OpenAI-compatible vision content
// blocks; URL is path-only (e.g. "/v1/artifacts/{token}.png") and the agent
// prefixes the configured PublicBaseURL when injecting the synthetic user
// message. DocID and Ref echo the underlying chunk marker for UI display.
type ImageRef struct {
	DocID string `json:"doc_id"`
	Ref   string `json:"ref"`
	URL   string `json:"url"`
}

// toFunctionTool converts a Tool to an OpenAI function tool parameter.
func toFunctionTool(t Tool) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        t.Name(),
			Description: openai.String(t.Description()),
			Parameters:  t.Parameters(),
		},
	}
}
