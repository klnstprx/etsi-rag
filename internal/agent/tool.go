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
// external references (weather, date, calculator) leave this nil.
type ToolResult struct {
	Text      string
	Citations []Citation
}

// Citation is a tool-agnostic reference to an external resource (document,
// webpage, PDF). Both semantic search and web search produce these.
type Citation struct {
	Title string
	URL   string // may be empty when no URL is known
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
