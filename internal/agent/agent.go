package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/llm"
	"etsi-rag/internal/session"
	"etsi-rag/internal/tokencount"
)

// ChatResult is the result of an agent conversation turn.
type ChatResult struct {
	Response  string
	Citations []Citation
}

// Agent orchestrates LLM tool-calling loops with session management.
// Fields are set once by NewAgent and never mutated — live config reload
// constructs a fresh Agent and swaps the pointer at the owning Handler.
type Agent struct {
	client              *llm.ChatClient
	tools               []Tool
	system              string
	maxIter             int
	counter             tokencount.Counter
	contextWindow       int
	toolOutputMaxTokens int
	sessionStore        *session.Store
	logger              zerolog.Logger
}

// NewAgent creates a new agent with the given configuration.
// counter and contextWindow enable tool output size guards. If either is nil/zero,
// the guard is disabled and a warning is logged. toolOutputMaxTokens overrides the
// default (contextWindow/4) ceiling when > 0.
func NewAgent(client *llm.ChatClient, tools []Tool, system string, maxIter int, counter tokencount.Counter, contextWindow, toolOutputMaxTokens int, sessionStore *session.Store, logger zerolog.Logger) *Agent {
	if counter == nil || contextWindow <= 0 {
		logger.Warn().Msg("agent tool output guards disabled (no token counter or context window)")
	}
	return &Agent{
		client:              client,
		tools:               tools,
		system:              system,
		maxIter:             maxIter,
		counter:             counter,
		contextWindow:       contextWindow,
		toolOutputMaxTokens: toolOutputMaxTokens,
		sessionStore:        sessionStore,
		logger:              logger,
	}
}

// Chat handles a session-managed conversation turn.
// It loads session history, runs the tool-calling loop, persists all messages,
// and returns the response with collected sources.
// events may be nil; if non-nil, ToolEvent values are sent during tool execution.
func (a *Agent) Chat(ctx context.Context, sessionID string, userMessage string, events chan<- ToolEvent) (*ChatResult, error) {
	client := a.client
	if client == nil {
		return nil, fmt.Errorf("agent not configured")
	}

	// Load session
	sess := a.sessionStore.Get(sessionID)
	if sess == nil {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	system := a.system
	tools := a.tools
	maxIter := a.maxIter

	// Build messages: system + session history + new user message
	history := sess.GetMessages()
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(history)+2)
	messages = append(messages, openai.ChatCompletionMessageParamUnion{
		OfSystem: &openai.ChatCompletionSystemMessageParam{
			Content: openai.ChatCompletionSystemMessageParamContentUnion{
				OfString: openai.String(system),
			},
		},
	})
	messages = append(messages, history...)

	userMsg := openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(userMessage),
			},
		},
	}
	messages = append(messages, userMsg)

	var citations []Citation

	// Run tool loop
	response, turnMessages, err := a.runToolLoop(ctx, client, messages, tools, maxIter, &citations, events, nil)
	if err != nil {
		return nil, err
	}

	// Persist all turn messages to session: user message + intermediate + final
	sess.AddMessage(userMsg)
	for _, msg := range turnMessages {
		sess.AddMessage(msg)
	}

	return &ChatResult{Response: response, Citations: citations}, nil
}

// ChatStreamOutcome is the terminal value emitted on ChatStream's outcome channel
// once the agent goroutine finishes. Result is nil when Err is non-nil.
type ChatStreamOutcome struct {
	Result *ChatResult
	Err    error
}

// ChatStream is like Chat but streams the assistant's content token-by-token to
// the returned tokens channel. The outcome channel emits exactly one value when
// the agent goroutine finishes (success or error) and is then closed. tokens is
// closed when streaming finishes; events is owned by the caller and is NOT
// closed here. events may be nil; if non-nil, ToolEvent values are sent during
// tool execution.
func (a *Agent) ChatStream(ctx context.Context, sessionID string, userMessage string, events chan<- ToolEvent) (<-chan string, <-chan ChatStreamOutcome) {
	tokens := make(chan string, 64)
	outcome := make(chan ChatStreamOutcome, 1)

	go func() {
		defer close(tokens)
		defer close(outcome)

		client := a.client
		if client == nil {
			outcome <- ChatStreamOutcome{Err: fmt.Errorf("agent not configured")}
			return
		}

		sess := a.sessionStore.Get(sessionID)
		if sess == nil {
			outcome <- ChatStreamOutcome{Err: fmt.Errorf("session not found: %s", sessionID)}
			return
		}

		system := a.system
		tools := a.tools
		maxIter := a.maxIter

		history := sess.GetMessages()
		messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(history)+2)
		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(system),
				},
			},
		})
		messages = append(messages, history...)

		userMsg := openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(userMessage),
				},
			},
		}
		messages = append(messages, userMsg)

		var citations []Citation

		response, turnMessages, err := a.runToolLoop(ctx, client, messages, tools, maxIter, &citations, events, tokens)
		if err != nil {
			outcome <- ChatStreamOutcome{Err: err}
			return
		}

		sess.AddMessage(userMsg)
		for _, msg := range turnMessages {
			sess.AddMessage(msg)
		}

		outcome <- ChatStreamOutcome{Result: &ChatResult{Response: response, Citations: citations}}
	}()

	return tokens, outcome
}

// ChatPassthrough handles a stateless conversation (client provides full history).
// No session management — the client is responsible for message history.
// events may be nil; if non-nil, ToolEvent values are sent during tool execution.
func (a *Agent) ChatPassthrough(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion, events chan<- ToolEvent) (*ChatResult, error) {
	client := a.client
	if client == nil {
		return nil, fmt.Errorf("agent not configured")
	}

	tools := a.tools
	maxIter := a.maxIter
	system := a.system

	// Prepend system prompt
	full := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages)+1)
	full = append(full, openai.ChatCompletionMessageParamUnion{
		OfSystem: &openai.ChatCompletionSystemMessageParam{
			Content: openai.ChatCompletionSystemMessageParamContentUnion{
				OfString: openai.String(system),
			},
		},
	})
	full = append(full, messages...)

	var citations []Citation

	response, _, err := a.runToolLoop(ctx, client, full, tools, maxIter, &citations, events, nil)
	if err != nil {
		return nil, err
	}

	return &ChatResult{Response: response, Citations: citations}, nil
}

// runToolLoop executes the tool-calling loop and returns the final text response
// plus all intermediate messages (assistant+tool messages, excluding the initial messages).
// events may be nil; if non-nil, ToolEvent values are sent for each tool execution.
// tokens may be nil; if non-nil, the streaming LLM path is used and assistant content
// deltas are forwarded onto it as they arrive. Ownership of `tokens` (closing) stays
// with the caller.
func (a *Agent) runToolLoop(
	ctx context.Context,
	client *llm.ChatClient,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []Tool,
	maxIter int,
	citations *[]Citation,
	events chan<- ToolEvent,
	tokens chan<- string,
) (string, []openai.ChatCompletionMessageParamUnion, error) {
	toolParams := make([]openai.ChatCompletionToolParam, len(tools))
	for i, t := range tools {
		toolParams[i] = toFunctionTool(t)
	}

	var turnMessages []openai.ChatCompletionMessageParamUnion

	for range maxIter {
		var resp *openai.ChatCompletion
		var err error
		if tokens != nil {
			deltas, resCh := client.ChatStreamWithTools(ctx, messages, toolParams)
			for delta := range deltas {
				tokens <- delta
			}
			res := <-resCh
			if res.Err != nil {
				return "", nil, fmt.Errorf("agent chat: %w", res.Err)
			}
			resp = res.Completion
		} else {
			resp, err = client.ChatWithTools(ctx, messages, toolParams)
			if err != nil {
				return "", nil, fmt.Errorf("agent chat: %w", err)
			}
		}

		if len(resp.Choices) == 0 {
			return "", nil, fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]

		// No tool calls → final response
		if len(choice.Message.ToolCalls) == 0 {
			finalMsg := openai.ChatCompletionMessageParamUnion{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(choice.Message.Content),
					},
				},
			}
			turnMessages = append(turnMessages, finalMsg)
			return choice.Message.Content, turnMessages, nil
		}

		// Append assistant message with tool calls
		assistantMsg := openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCallsToParams(choice.Message.ToolCalls),
			},
		}
		messages = append(messages, assistantMsg)
		turnMessages = append(turnMessages, assistantMsg)

		// Execute each tool call
		for _, tc := range choice.Message.ToolCalls {
			if events != nil {
				events <- ToolEvent{
					Type:   "start",
					CallID: tc.ID,
					Name:   tc.Function.Name,
					Args:   tc.Function.Arguments,
				}
			}

			execStart := time.Now()
			toolRes, err := a.executeTool(ctx, tc.Function.Name, tc.Function.Arguments)
			execDuration := time.Since(execStart)
			toolErr := err != nil

			var resultText string
			var callCitations []Citation
			if err != nil {
				a.logger.Warn().Err(err).Str("tool", tc.Function.Name).Msg("tool execution failed")
				resultText = fmt.Sprintf("Error: %s", err.Error())
			} else {
				resultText = toolRes.Text
				callCitations = toolRes.Citations
				if citations != nil && len(callCitations) > 0 {
					*citations = append(*citations, callCitations...)
				}
			}

			resultText = a.checkToolResult(ctx, tc.Function.Name, resultText)

			if events != nil {
				events <- ToolEvent{
					Type:      "done",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Result:    resultText,
					Duration:  execDuration,
					Error:     toolErr,
					Citations: callCitations,
				}
			}

			a.logger.Debug().
				Str("tool", tc.Function.Name).
				Str("args", tc.Function.Arguments).
				Int("result_len", len(resultText)).
				Int("citations", len(callCitations)).
				Dur("duration", execDuration).
				Msg("tool executed")

			toolMsg := openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: tc.ID,
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(resultText),
					},
				},
			}
			messages = append(messages, toolMsg)
			turnMessages = append(turnMessages, toolMsg)
		}
	}

	return "", nil, fmt.Errorf("agent exceeded max tool iterations (%d)", maxIter)
}

func (a *Agent) executeTool(ctx context.Context, name string, argsJSON string) (ToolResult, error) {
	for _, t := range a.tools {
		if t.Name() == name {
			return t.Execute(ctx, argsJSON)
		}
	}
	return ToolResult{}, fmt.Errorf("unknown tool: %s", name)
}

// checkToolResult validates that a tool result fits within the context budget.
// If the result is too large, it returns an error message instead so the LLM
// can self-correct (e.g., refine a query). Returns the result unchanged if
// within budget or if guards are disabled (no counter/context window).
//
// The ceiling is either an explicitly configured toolOutputMaxTokens (>0) or
// an auto-derived default of contextWindow/4 with a 256-token floor.
func (a *Agent) checkToolResult(ctx context.Context, toolName string, result string) string {
	counter := a.counter
	contextWindow := a.contextWindow
	if counter == nil || contextWindow <= 0 {
		return result
	}

	maxTokens := a.toolOutputMaxTokens
	if maxTokens <= 0 {
		maxTokens = max(contextWindow/4, 256)
	}

	tokens, err := counter.Count(ctx, result)
	if err != nil {
		a.logger.Warn().Err(err).Str("tool", toolName).Msg("failed to count tool result tokens")
		return result
	}

	if tokens > maxTokens {
		a.logger.Warn().
			Str("tool", toolName).
			Int("tokens", tokens).
			Int("budget", maxTokens).
			Msg("tool result exceeded token budget")
		return fmt.Sprintf("Tool result too large (%d tokens, budget %d). Refine your query or use more specific parameters.", tokens, maxTokens)
	}

	return result
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
