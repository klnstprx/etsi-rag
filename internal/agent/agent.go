package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/llm"
	"etsi-rag/internal/session"
	"etsi-rag/internal/tokencount"
)

// minToolResultTokens floors the derived tool-result cap so an over-subscribed
// budget can't reject every result outright.
const minToolResultTokens = 256

// ChatResult is the result of an agent conversation turn.
type ChatResult struct {
	Response  string
	Citations []Citation
	Images    []ImageRef
	ToolCalls []ToolCallRecord
}

// ToolCallRecord captures one tool invocation inside the agent loop. Used by
// downstream consumers (eval harness, observability) that need to know which
// tools fired in a turn — distinct from ToolEvent (which is the streaming
// channel shape) so we don't bind the persisted record to the live event
// schema. Args is the raw JSON arguments string from the LLM (truncated by
// the LLM's own tool-calling format if very long).
type ToolCallRecord struct {
	Name       string `json:"name"`
	Args       string `json:"args,omitempty"`
	Error      bool   `json:"error,omitempty"`
	DurationMS int64  `json:"duration_ms"`
}

// VisionGetter returns whether the configured chat backend supports vision
// content blocks and the absolute public base URL prefixed onto path-only
// artifact URLs (e.g. "http://host:port"). Wired through a getter rather
// than cached fields so admin-UI toggles take effect without a restart.
type VisionGetter func() (supportsVision bool, baseURL string)

// Agent orchestrates LLM tool-calling loops with session management.
// Fields are set once by NewAgent and never mutated — live config reload
// constructs a fresh Agent and swaps the pointer at the owning Handler.
type Agent struct {
	client        *llm.ChatClient
	tools         []Tool
	system        string
	maxIter       int
	counter       tokencount.Counter
	toolResultCap int // max tokens per tool result; 0 disables the guard
	sessionStore  *session.Store
	logger        zerolog.Logger
	vision        VisionGetter
}

// NewAgent creates a new agent with the given configuration.
// counter and budget enable the per-tool-result size guard. If counter is nil
// or budget.ContextWindow <= 0, the guard is disabled and a warning is logged.
// toolOutputMaxTokens is an explicit operator override: when > 0 it is used as
// the cap directly; otherwise the cap is derived from budget.ToolPool() with a
// minToolResultTokens floor.
func NewAgent(client *llm.ChatClient, tools []Tool, system string, maxIter int, counter tokencount.Counter, budget Budget, toolOutputMaxTokens int, sessionStore *session.Store, logger zerolog.Logger) *Agent {
	var toolResultCap int
	switch {
	case counter == nil || budget.ContextWindow <= 0:
		logger.Warn().Msg("agent tool output guards disabled (no token counter or context window)")
	case toolOutputMaxTokens > 0:
		toolResultCap = toolOutputMaxTokens
	default:
		toolResultCap = max(budget.ToolPool(), minToolResultTokens)
	}
	return &Agent{
		client:        client,
		tools:         tools,
		system:        system,
		maxIter:       maxIter,
		counter:       counter,
		toolResultCap: toolResultCap,
		sessionStore:  sessionStore,
		logger:        logger,
	}
}

// SetVisionGetter installs a getter for vision capability + public base URL.
// When unset (default), no synthetic user message is injected and tool images
// are not surfaced to the LLM. The getter is called per tool call to honour
// admin toggles without restart.
func (a *Agent) SetVisionGetter(g VisionGetter) { a.vision = g }

// Chat handles a session-managed conversation turn.
// It loads session history, runs the tool-calling loop, persists all messages,
// and returns the response with collected sources.
// events may be nil; if non-nil, ToolEvent values are sent during tool execution.
func (a *Agent) Chat(ctx context.Context, sessionID string, userMessage string, events chan<- ToolEvent) (*ChatResult, error) {
	return a.turn(ctx, sessionID, userMessage, events, nil)
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

		res, err := a.turn(ctx, sessionID, userMessage, events, tokens)
		outcome <- ChatStreamOutcome{Result: res, Err: err}
	}()

	return tokens, outcome
}

// turn runs one session-managed conversation turn: system prompt + history +
// user message through the tool loop, then persists the user message and every
// message the turn produced. A nil tokens channel selects the blocking LLM call.
func (a *Agent) turn(ctx context.Context, sessionID, userMessage string, events chan<- ToolEvent, tokens chan<- string) (*ChatResult, error) {
	if a.client == nil {
		return nil, errors.New("agent not configured")
	}
	sess := a.sessionStore.Get(sessionID)
	if sess == nil {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	history := sess.GetMessages()
	userMsg := openai.UserMessage(userMessage)
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(history)+2)
	messages = append(messages, openai.SystemMessage(a.system))
	messages = append(messages, history...)
	messages = append(messages, userMsg)

	res, turnMessages, err := a.runToolLoop(ctx, messages, events, tokens)
	if err != nil {
		return nil, err
	}

	sess.AddMessage(userMsg)
	for _, msg := range turnMessages {
		sess.AddMessage(msg)
	}
	return res, nil
}

// runToolLoop executes the tool-calling loop and returns the turn's result
// plus all intermediate messages (assistant+tool messages, excluding the initial messages).
// events may be nil; if non-nil, ToolEvent values are sent for each tool execution.
// tokens may be nil; if non-nil, assistant content deltas are forwarded onto it.
// Ownership of `tokens` (closing) stays with the caller.
func (a *Agent) runToolLoop(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	events chan<- ToolEvent,
	tokens chan<- string,
) (*ChatResult, []openai.ChatCompletionMessageParamUnion, error) {
	toolParams := make([]openai.ChatCompletionToolParam, len(a.tools))
	for i, t := range a.tools {
		toolParams[i] = toFunctionTool(t)
	}

	var res ChatResult
	var turnMessages []openai.ChatCompletionMessageParamUnion

	// Tools dedupe within a single call; across calls the same document (and
	// its page images) can resurface, so the accumulated UI-facing lists are
	// deduped here.
	seenCitations := make(map[string]struct{})
	seenImages := make(map[string]struct{})

	for range a.maxIter {
		resp, err := a.complete(ctx, messages, toolParams, tokens)
		if err != nil {
			return nil, nil, fmt.Errorf("agent chat: %w", err)
		}
		if len(resp.Choices) == 0 {
			return nil, nil, fmt.Errorf("no choices in response")
		}

		choice := resp.Choices[0]

		// No tool calls → final response
		if len(choice.Message.ToolCalls) == 0 {
			turnMessages = append(turnMessages, openai.AssistantMessage(choice.Message.Content))
			res.Response = choice.Message.Content
			return &res, turnMessages, nil
		}

		assistantMsg := openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				ToolCalls: toolCallsToParams(choice.Message.ToolCalls),
			},
		}
		messages = append(messages, assistantMsg)
		turnMessages = append(turnMessages, assistantMsg)

		for _, tc := range choice.Message.ToolCalls {
			call := a.callTool(ctx, tc, events)
			res.ToolCalls = append(res.ToolCalls, call.record)
			for _, c := range call.citations {
				key := c.URL
				if key == "" {
					key = c.Title
				}
				if _, dup := seenCitations[key]; dup {
					continue
				}
				seenCitations[key] = struct{}{}
				res.Citations = append(res.Citations, c)
			}
			for _, img := range call.attached {
				key := img.DocID + "\x00" + img.Ref
				if _, dup := seenImages[key]; dup {
					continue
				}
				seenImages[key] = struct{}{}
				res.Images = append(res.Images, img)
			}
			messages = append(messages, call.messages...)
			turnMessages = append(turnMessages, call.messages...)
		}
	}

	return nil, nil, fmt.Errorf("agent exceeded max tool iterations (%d)", a.maxIter)
}

// complete makes one LLM call, streaming content deltas onto tokens when it is
// non-nil.
func (a *Agent) complete(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	toolParams []openai.ChatCompletionToolParam,
	tokens chan<- string,
) (*openai.ChatCompletion, error) {
	if tokens == nil {
		return a.client.ChatWithTools(ctx, messages, toolParams)
	}
	deltas, results := a.client.ChatStreamWithTools(ctx, messages, toolParams)
	for delta := range deltas {
		tokens <- delta
	}
	res := <-results
	return res.Completion, res.Err
}

// toolCall is the outcome of one executed tool call.
type toolCall struct {
	record    ToolCallRecord
	citations []Citation // empty when the tool failed
	attached  []ImageRef // images forwarded to the LLM
	// messages is the tool message, followed by a synthetic user message
	// carrying the attached images when there are any.
	messages []openai.ChatCompletionMessageParamUnion
}

// callTool executes one tool call, applies the result-size guard and vision
// gating, and emits its start/done events.
func (a *Agent) callTool(ctx context.Context, tc openai.ChatCompletionMessageToolCall, events chan<- ToolEvent) toolCall {
	name, args := tc.Function.Name, tc.Function.Arguments
	if events != nil {
		events <- ToolEvent{Type: "start", CallID: tc.ID, Name: name, Args: args}
	}

	start := time.Now()
	toolRes, err := a.executeTool(ctx, name, args)
	duration := time.Since(start)
	toolErr := err != nil

	var resultText string
	var citations []Citation
	if toolErr {
		a.logger.Warn().Err(err).Str("tool", name).Msg("tool execution failed")
		resultText = fmt.Sprintf("Error: %s", err.Error())
	} else {
		resultText = toolRes.Text
		citations = toolRes.Citations
	}
	resultText = a.checkToolResult(ctx, name, resultText)

	var attached []ImageRef
	var baseURL string
	if !toolErr {
		attached, baseURL = a.visionImages(name, toolRes.Images)
	}

	if events != nil {
		events <- ToolEvent{
			Type:           "done",
			CallID:         tc.ID,
			Name:           name,
			Args:           args,
			Result:         resultText,
			Duration:       duration,
			Error:          toolErr,
			Citations:      citations,
			Images:         toolRes.Images,
			VisionAttached: len(attached) > 0,
		}
	}

	a.logger.Debug().
		Str("tool", name).
		Str("args", args).
		Int("result_len", len(resultText)).
		Int("citations", len(citations)).
		Int("tool_images", len(toolRes.Images)).
		Int("attached_images", len(attached)).
		Dur("duration", duration).
		Msg("tool executed")

	msgs := []openai.ChatCompletionMessageParamUnion{openai.ToolMessage(resultText, tc.ID)}
	// openai-go v1.12.0's tool-message Content union is text-only, so images
	// ride in a synthetic user turn right after the tool's text.
	if len(attached) > 0 {
		msgs = append(msgs, buildImageUserMessage(baseURL, attached))
	}

	return toolCall{
		record: ToolCallRecord{
			Name:       name,
			Args:       args,
			Error:      toolErr,
			DurationMS: duration.Milliseconds(),
		},
		citations: citations,
		attached:  attached,
		messages:  msgs,
	}
}

// visionImages returns the tool images to forward to the LLM and the base URL
// to prefix them with. Withheld images are logged with the reason so a missed
// admin toggle or empty config doesn't silently degrade retrieval quality.
func (a *Agent) visionImages(tool string, images []ImageRef) ([]ImageRef, string) {
	if a.vision == nil {
		if len(images) > 0 {
			a.logger.Warn().
				Str("tool", tool).
				Int("dropped_images", len(images)).
				Msg("tool returned images but agent has no VisionGetter installed — image attachment disabled")
		}
		return nil, ""
	}

	vision, baseURL := a.vision()
	switch {
	case len(images) == 0:
		// Common for HTML-only hits or when no top-K chunk carries a marker;
		// not actionable in itself.
		a.logger.Debug().
			Str("tool", tool).
			Bool("vision_enabled", vision && baseURL != "").
			Msg("no images returned by tool")
		return nil, ""
	case vision && baseURL != "":
		return images, baseURL
	case vision:
		a.logger.Warn().
			Str("tool", tool).
			Int("dropped_images", len(images)).
			Msg("backend supports vision but [agent].public_base_url is empty — image attachment disabled")
	default:
		a.logger.Warn().
			Str("tool", tool).
			Int("dropped_images", len(images)).
			Msg("tool returned images but agent.vision.enabled=false — image attachment disabled")
	}
	return nil, ""
}

func (a *Agent) executeTool(ctx context.Context, name string, argsJSON string) (ToolResult, error) {
	for _, t := range a.tools {
		if t.Name() == name {
			return t.Execute(ctx, argsJSON)
		}
	}
	return ToolResult{}, fmt.Errorf("unknown tool: %s", name)
}

// checkToolResult validates that a tool result fits within toolResultCap.
// If the result is too large, it returns an error message instead so the LLM
// can self-correct (e.g., refine a query). Returns the result unchanged if
// within budget or if the guard is disabled.
func (a *Agent) checkToolResult(ctx context.Context, toolName string, result string) string {
	if a.toolResultCap == 0 {
		return result
	}

	tokens, err := a.counter.Count(ctx, result)
	if err != nil {
		a.logger.Warn().Err(err).Str("tool", toolName).Msg("failed to count tool result tokens")
		return result
	}

	if tokens > a.toolResultCap {
		a.logger.Warn().
			Str("tool", toolName).
			Int("tokens", tokens).
			Int("budget", a.toolResultCap).
			Msg("tool result exceeded token budget")
		return fmt.Sprintf("Tool result too large (%d tokens, budget %d). Refine your query or use more specific parameters.", tokens, a.toolResultCap)
	}

	return result
}

// buildImageUserMessage builds a {role:user, content:[text+image_url parts]}
// message from a list of image refs, prefixing baseURL onto each path-only
// URL so the LLM backend can fetch it. The leading text part is a brief
// instruction so the model knows the images are tool-derived context.
func buildImageUserMessage(baseURL string, images []ImageRef) openai.ChatCompletionMessageParamUnion {
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(images)+1)
	parts = append(parts, openai.ChatCompletionContentPartUnionParam{
		OfText: &openai.ChatCompletionContentPartTextParam{
			Text: "The previous tool returned the following page images for visual context:",
		},
	})
	for _, img := range images {
		parts = append(parts, openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL: baseURL + img.URL,
				},
			},
		})
	}
	return openai.UserMessage(parts)
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
