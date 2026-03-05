package agent

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/llm"
)

const maxToolIterations = 5

// Agent orchestrates LLM tool-calling loops.
type Agent struct {
	client *llm.Client
	tools  []Tool
	system string
	logger zerolog.Logger
}

// NewAgent creates a new agent with the given tools.
func NewAgent(client *llm.Client, tools []Tool, system string, logger zerolog.Logger) *Agent {
	return &Agent{
		client: client,
		tools:  tools,
		system: system,
		logger: logger,
	}
}

// Run executes the agent loop for a user message, returning the final text response.
func (a *Agent) Run(ctx context.Context, userMessage string) (string, error) {
	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(a.system),
				},
			},
		},
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(userMessage),
				},
			},
		},
	}

	toolParams := make([]openai.ChatCompletionToolParam, len(a.tools))
	for i, t := range a.tools {
		toolParams[i] = t.toFunctionTool()
	}

	for i := 0; i < maxToolIterations; i++ {
		resp, err := a.client.ChatWithTools(ctx, messages, toolParams)
		if err != nil {
			return "", fmt.Errorf("agent chat: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]

		// If no tool calls, return the text content
		if len(choice.Message.ToolCalls) == 0 {
			return choice.Message.Content, nil
		}

		// Append assistant message with tool calls
		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCallsToParams(choice.Message.ToolCalls),
			},
		})

		// Execute each tool call
		for _, tc := range choice.Message.ToolCalls {
			result, err := a.executeTool(ctx, tc.Function.Name, tc.Function.Arguments)
			if err != nil {
				a.logger.Warn().Err(err).Str("tool", tc.Function.Name).Msg("tool execution failed")
				result = fmt.Sprintf("Error: %s", err.Error())
			}

			a.logger.Debug().
				Str("tool", tc.Function.Name).
				Str("args", tc.Function.Arguments).
				Int("result_len", len(result)).
				Msg("tool executed")

			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: tc.ID,
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(result),
					},
				},
			})
		}
	}

	return "", fmt.Errorf("agent exceeded max tool iterations (%d)", maxToolIterations)
}

func (a *Agent) executeTool(ctx context.Context, name string, argsJSON string) (string, error) {
	for _, t := range a.tools {
		if t.Name == name {
			return t.Execute(ctx, argsJSON)
		}
	}
	return "", fmt.Errorf("unknown tool: %s", name)
}

// RunStream executes the agent loop and streams the final response.
func (a *Agent) RunStream(ctx context.Context, userMessage string) (<-chan string, error) {
	// For streaming, we first run the tool loop, then stream the final response
	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(a.system),
				},
			},
		},
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(userMessage),
				},
			},
		},
	}

	toolParams := make([]openai.ChatCompletionToolParam, len(a.tools))
	for i, t := range a.tools {
		toolParams[i] = t.toFunctionTool()
	}

	// Tool-calling loop (non-streaming)
	for i := 0; i < maxToolIterations; i++ {
		resp, err := a.client.ChatWithTools(ctx, messages, toolParams)
		if err != nil {
			return nil, fmt.Errorf("agent chat: %w", err)
		}

		if len(resp.Choices) == 0 {
			return nil, fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]

		if len(choice.Message.ToolCalls) == 0 {
			// Final response — stream it
			tokens := make(chan string, 64)
			go func() {
				defer close(tokens)
				// Send the complete text as a single token since this came from
				// a non-streaming call
				if choice.Message.Content != "" {
					tokens <- choice.Message.Content
				}
			}()
			return tokens, nil
		}

		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCallsToParams(choice.Message.ToolCalls),
			},
		})

		for _, tc := range choice.Message.ToolCalls {
			result, err := a.executeTool(ctx, tc.Function.Name, tc.Function.Arguments)
			if err != nil {
				result = fmt.Sprintf("Error: %s", err.Error())
			}

			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: tc.ID,
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(result),
					},
				},
			})
		}
	}

	return nil, fmt.Errorf("agent exceeded max tool iterations (%d)", maxToolIterations)
}

// toolCallsToParams converts response tool calls to param tool calls.
func toolCallsToParams(calls []openai.ChatCompletionMessageToolCall) []openai.ChatCompletionMessageToolCallParam {
	params := make([]openai.ChatCompletionMessageToolCallParam, len(calls))
	for i, tc := range calls {
		params[i] = openai.ChatCompletionMessageToolCallParam{
			ID: tc.ID,
			Function: openai.ChatCompletionMessageToolCallFunctionParam{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return params
}
