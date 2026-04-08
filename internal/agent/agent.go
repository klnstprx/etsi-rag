package agent

import (
	"context"
	"fmt"
	"sync"

	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/session"
	"etsi-rag/internal/tokencount"
)

// ChatResult is the result of an agent conversation turn.
type ChatResult struct {
	Response string
	Sources  []domain.SourceReference
}

// Agent orchestrates LLM tool-calling loops with session management.
type Agent struct {
	mu            sync.RWMutex
	client        *llm.ChatClient
	tools         []Tool
	system        string
	maxIter       int
	counter       tokencount.Counter
	contextWindow int
	sessionStore  *session.Store
	logger        zerolog.Logger
}

// NewAgent creates a new agent with the given configuration.
// counter and contextWindow enable tool output size guards. If either is nil/zero,
// the guard is disabled and a warning is logged.
func NewAgent(client *llm.ChatClient, tools []Tool, system string, maxIter int, counter tokencount.Counter, contextWindow int, sessionStore *session.Store, logger zerolog.Logger) *Agent {
	if counter == nil || contextWindow <= 0 {
		logger.Warn().Msg("agent tool output guards disabled (no token counter or context window)")
	}
	return &Agent{
		client:        client,
		tools:         tools,
		system:        system,
		maxIter:       maxIter,
		counter:       counter,
		contextWindow: contextWindow,
		sessionStore:  sessionStore,
		logger:        logger,
	}
}

// SetClient swaps the chat client (thread-safe).
func (a *Agent) SetClient(c *llm.ChatClient) {
	a.mu.Lock()
	a.client = c
	a.mu.Unlock()
}

// SetSystem swaps the system prompt (thread-safe).
func (a *Agent) SetSystem(s string) {
	a.mu.Lock()
	a.system = s
	a.mu.Unlock()
}

// SetTools swaps the tool list (thread-safe).
func (a *Agent) SetTools(t []Tool) {
	a.mu.Lock()
	a.tools = t
	a.mu.Unlock()
}

// SetMaxIterations swaps the max iterations (thread-safe).
func (a *Agent) SetMaxIterations(n int) {
	a.mu.Lock()
	a.maxIter = n
	a.mu.Unlock()
}

// SetCounter swaps the token counter (thread-safe).
func (a *Agent) SetCounter(c tokencount.Counter) {
	a.mu.Lock()
	a.counter = c
	a.mu.Unlock()
}

// SetContextWindow swaps the context window size (thread-safe).
func (a *Agent) SetContextWindow(n int) {
	a.mu.Lock()
	a.contextWindow = n
	a.mu.Unlock()
}

func (a *Agent) getClient() *llm.ChatClient {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.client
}

func (a *Agent) getSystem() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.system
}

func (a *Agent) getTools() []Tool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.tools
}

func (a *Agent) getCounter() tokencount.Counter {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.counter
}

func (a *Agent) getContextWindow() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.contextWindow
}

func (a *Agent) getMaxIter() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.maxIter
}

// Chat handles a session-managed conversation turn.
// It loads session history, runs the tool-calling loop, persists all messages,
// and returns the response with collected sources.
func (a *Agent) Chat(ctx context.Context, sessionID string, userMessage string) (*ChatResult, error) {
	client := a.getClient()
	if client == nil {
		return nil, fmt.Errorf("agent not configured")
	}

	// Load session
	sess := a.sessionStore.Get(sessionID)
	if sess == nil {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	system := a.getSystem()
	tools := a.getTools()
	maxIter := a.getMaxIter()

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

	// Reset per-turn sources
	var sources []domain.SourceReference

	// Run tool loop
	response, turnMessages, err := a.runToolLoop(ctx, client, messages, tools, maxIter, &sources)
	if err != nil {
		return nil, err
	}

	// Persist all turn messages to session: user message + intermediate + final
	sess.AddMessage(userMsg)
	for _, msg := range turnMessages {
		sess.AddMessage(msg)
	}

	return &ChatResult{Response: response, Sources: sources}, nil
}

// ChatStream handles a session-managed conversation turn with streaming final response.
func (a *Agent) ChatStream(ctx context.Context, sessionID string, userMessage string) (<-chan string, *ChatResult, error) {
	// For now, run the full turn, then stream the result as a single chunk.
	// True streaming of the final LLM call can be added later.
	result, err := a.Chat(ctx, sessionID, userMessage)
	if err != nil {
		return nil, nil, err
	}

	tokens := make(chan string, 1)
	go func() {
		defer close(tokens)
		if result.Response != "" {
			tokens <- result.Response
		}
	}()

	return tokens, result, nil
}

// ChatPassthrough handles a stateless conversation (client provides full history).
// No session management — the client is responsible for message history.
func (a *Agent) ChatPassthrough(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (*ChatResult, error) {
	client := a.getClient()
	if client == nil {
		return nil, fmt.Errorf("agent not configured")
	}

	tools := a.getTools()
	maxIter := a.getMaxIter()
	system := a.getSystem()

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

	var sources []domain.SourceReference

	response, _, err := a.runToolLoop(ctx, client, full, tools, maxIter, &sources)
	if err != nil {
		return nil, err
	}

	return &ChatResult{Response: response, Sources: sources}, nil
}

// runToolLoop executes the tool-calling loop and returns the final text response
// plus all intermediate messages (assistant+tool messages, excluding the initial messages).
func (a *Agent) runToolLoop(
	ctx context.Context,
	client *llm.ChatClient,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []Tool,
	maxIter int,
	sources *[]domain.SourceReference,
) (string, []openai.ChatCompletionMessageParamUnion, error) {
	toolParams := make([]openai.ChatCompletionToolParam, len(tools))
	for i, t := range tools {
		toolParams[i] = t.toFunctionTool()
	}

	var turnMessages []openai.ChatCompletionMessageParamUnion

	for i := 0; i < maxIter; i++ {
		resp, err := client.ChatWithTools(ctx, messages, toolParams)
		if err != nil {
			return "", nil, fmt.Errorf("agent chat: %w", err)
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
			result, err := a.executeTool(ctx, tc.Function.Name, tc.Function.Arguments, sources)
			if err != nil {
				a.logger.Warn().Err(err).Str("tool", tc.Function.Name).Msg("tool execution failed")
				result = fmt.Sprintf("Error: %s", err.Error())
			}

			result = a.checkToolResult(ctx, tc.Function.Name, result)

			a.logger.Debug().
				Str("tool", tc.Function.Name).
				Str("args", tc.Function.Arguments).
				Int("result_len", len(result)).
				Msg("tool executed")

			toolMsg := openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: tc.ID,
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(result),
					},
				},
			}
			messages = append(messages, toolMsg)
			turnMessages = append(turnMessages, toolMsg)
		}
	}

	return "", nil, fmt.Errorf("agent exceeded max tool iterations (%d)", maxIter)
}

func (a *Agent) executeTool(ctx context.Context, name string, argsJSON string, sources *[]domain.SourceReference) (string, error) {
	for _, t := range a.tools {
		if t.Name == name {
			return t.Execute(ctx, argsJSON, sources)
		}
	}
	return "", fmt.Errorf("unknown tool: %s", name)
}

// checkToolResult validates that a tool result fits within the context budget.
// If the result is too large, it returns an error message instead so the LLM
// can self-correct (e.g., refine a query). Returns the result unchanged if
// within budget or if guards are disabled (no counter/context window).
func (a *Agent) checkToolResult(ctx context.Context, toolName string, result string) string {
	counter := a.getCounter()
	contextWindow := a.getContextWindow()
	if counter == nil || contextWindow <= 0 {
		return result
	}

	maxTokens := contextWindow / 4
	if maxTokens < 256 {
		maxTokens = 256
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
