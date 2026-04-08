package agent

import (
	"context"

	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
)

// Tool defines a callable tool for the agent.
// Execute receives a sources pointer so tools like rag_search can collect sources.
type Tool struct {
	Name        string
	Description string
	Parameters  openai.FunctionParameters
	Execute     func(ctx context.Context, args string, sources *[]domain.SourceReference) (string, error)
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
