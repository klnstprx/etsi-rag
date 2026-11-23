package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5/middleware"

	"etsi-rag/internal/agent"
)

// relayStream runs one agent turn through agent.RelayChat into the sink built
// by newSink, after setting the SSE and X-Session-ID headers. Relay errors are
// logged at debug only: they almost always mean the client disconnected, and
// the status line is already flushed.
func relayStream(w http.ResponseWriter, r *http.Request, ag *agent.Agent, sessionID, message string, newSink func(http.ResponseWriter, http.Flusher) agent.StreamSink) {
	flusher, err := prepareSSE(w)
	if err != nil {
		ErrInternal(err).WriteJSON(w, r)
		return
	}
	w.Header().Set(sessionHeader, sessionID)

	if err := agent.RelayChat(r.Context(), ag, sessionID, message, newSink(w, flusher)); err != nil {
		logger.Debug().
			Err(err).
			Str("path", r.URL.Path).
			Str("request_id", middleware.GetReqID(r.Context())).
			Msg("chat stream relay ended early (likely client disconnect)")
	}
}

// openaiSSESink implements agent.StreamSink for the public OpenAI-compatible
// streaming chat endpoint. It emits ONLY the standard `chat.completion.chunk`
// JSON stream terminated with `data: [DONE]\n\n` — no RAG extras.
//
// Tool lifecycle / citations are deliberately dropped here (OnToolEvent is a
// no-op): the OpenAI SDK's stream decoder yields a chunk for every `data:`
// line regardless of the SSE `event:` name, so emitting named events as extra
// SSE events surfaces them to the client as bogus empty-choices chunks (a
// naive `chunk.choices[0]` loop then crashes). Clients that want the RAG
// context use the proprietary POST /v1/agent/chat stream instead.
type openaiSSESink struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	chatID   string
	model    string
	created  int64
	roleSent bool
}

func newOpenAISSESink(w http.ResponseWriter, flusher http.Flusher, model string) *openaiSSESink {
	return &openaiSSESink{
		w:       w,
		flusher: flusher,
		chatID:  "chatcmpl-" + randomHex(12),
		model:   model,
		created: time.Now().Unix(),
	}
}

// streamChunkOut is the wire shape we emit for each `chat.completion.chunk`.
// We hand-roll this instead of reusing openai.ChatCompletionChunk because the
// SDK type carries read-side reflection metadata and forces a `constant`
// object discriminator that's awkward to set from the producer side.
type streamChunkOut struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []streamChoiceOut `json:"choices"`
}

type streamChoiceOut struct {
	Index        int         `json:"index"`
	Delta        streamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type streamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func (s *openaiSSESink) chunk(delta streamDelta, finish *string) streamChunkOut {
	return streamChunkOut{
		ID:      s.chatID,
		Object:  "chat.completion.chunk",
		Created: s.created,
		Model:   s.model,
		Choices: []streamChoiceOut{{Delta: delta, FinishReason: finish}},
	}
}

func (s *openaiSSESink) OnToken(tok string) error {
	if !s.roleSent {
		if err := writeSSE(s.w, s.flusher, "", s.chunk(streamDelta{Role: "assistant"}, nil)); err != nil {
			return err
		}
		s.roleSent = true
	}
	return writeSSE(s.w, s.flusher, "", s.chunk(streamDelta{Content: tok}, nil))
}

// OnToolEvent is a no-op for the OpenAI stream — see the type doc for why tool
// lifecycle cannot be surfaced as named SSE events without breaking SDK clients.
func (s *openaiSSESink) OnToolEvent(agent.ToolEvent) error { return nil }

func (s *openaiSSESink) OnEnd(_ *agent.ChatResult, agentErr error) error {
	if agentErr != nil {
		// OpenAI-idiomatic mid-stream error: an error object on the default
		// channel (the SDK raises on this), then [DONE].
		_ = writeSSE(s.w, s.flusher, "", map[string]any{
			"error": map[string]string{"message": agentErr.Error(), "type": "internal_error"},
		})
	} else {
		stop := "stop"
		if err := writeSSE(s.w, s.flusher, "", s.chunk(streamDelta{}, &stop)); err != nil {
			return err
		}
	}

	_, err := io.WriteString(s.w, "data: [DONE]\n\n")
	s.flusher.Flush()
	return err
}

// writeSSE writes payload as one JSON `data:` line and flushes. An empty
// event writes on the default channel, which OpenAI-spec chunks require.
func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string, payload any) error {
	buf, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal SSE event %q: %w", event, err)
	}
	if strings.ContainsRune(string(buf), '\n') {
		// A raw newline would split the event; json.Marshal never emits one.
		return fmt.Errorf("SSE payload for %q contains raw newline", event)
	}
	if event != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", event); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if _, err := w.Write(buf); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// prepareSSE sets headers and casts the writer to a Flusher. Returns the
// flusher or an error if the writer doesn't support streaming.
func prepareSSE(w http.ResponseWriter) (http.Flusher, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	return flusher, nil
}

func randomHex(n int) string {
	b := make([]byte, n)
	// crypto/rand.Read never returns an error — it crashes the program
	// irrecoverably on the (legacy-only) OS read failure rather than reporting
	// it — so there is no error branch to handle here.
	rand.Read(b)
	return hex.EncodeToString(b)
}
