package agent

import (
	"context"

	"github.com/openai/openai-go"
)

// Tool defines a callable tool for the agent.
type Tool struct {
	Name        string
	Description string
	Parameters  openai.FunctionParameters
	Execute     func(ctx context.Context, args string) (string, error)
}

// toFunctionTool converts a Tool to an OpenAI function tool parameter.
func (t Tool) toFunctionTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openai.String(t.Description),
			Parameters:  t.Parameters,
		},
	}
}
