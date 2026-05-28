package api

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5/middleware"

	"etsi-rag/internal/agent"
)

// AgentChatRequest is the proprietary agent-chat request: a single user turn.
// The session is carried in the X-Session-ID header (minted if absent). Unlike
// the OpenAI endpoint there is no message array or model field — the agent's
// backend/model is server-configured.
type AgentChatRequest struct {
	Message string `json:"message"`
	Stream  bool   `json:"stream,omitempty"`
}

// citationDTO is the wire shape for a citation. agent.Citation has no JSON
// tags (it's a UI-facing Go type), so we map it to clean lowercase keys here.
type citationDTO struct {
	Title string `json:"title,omitempty"`
	URL   string `json:"url,omitempty"`
}

func toCitationDTOs(cs []agent.Citation) []citationDTO {
	if len(cs) == 0 {
		return nil
	}
	out := make([]citationDTO, len(cs))
	for i, c := range cs {
		out[i] = citationDTO{Title: c.Title, URL: c.URL}
	}
	return out
}

// AgentChatResponse is the non-streaming proprietary response. It carries the
// full RAG context (citations, images, tool calls) that the OpenAI endpoint
// deliberately omits.
type AgentChatResponse struct {
	SessionID string                 `json:"session_id"`
	Content   string                 `json:"content"`
	Citations []citationDTO          `json:"citations,omitempty"`
	Images    []agent.ImageRef       `json:"images,omitempty"`
	ToolCalls []agent.ToolCallRecord `json:"tool_calls,omitempty"`
}

// AgentChat handles POST /v1/agent/chat — the proprietary RAG agent API used
// by the admin UI and the eval harness. Session-managed via X-Session-ID.
// When req.Stream is set it emits the proprietary JSON-SSE stream (event:
// token / tool / done); otherwise it returns a single AgentChatResponse.
func (h *Handler) AgentChat(w http.ResponseWriter, r *http.Request) {
	ag := h.agentHandle.Get()
	if ag == nil {
		ErrServiceUnavailable("agent not configured").WriteJSON(w, r)
		return
	}

	var req AgentChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ErrBadRequest("invalid request body").WriteJSON(w, r)
		return
	}
	if req.Message == "" {
		ErrBadRequest("message cannot be empty").WriteJSON(w, r)
		return
	}

	sessionIDHeader := r.Header.Get("X-Session-ID")
	var sessionIDPtr *string
	if sessionIDHeader != "" {
		sessionIDPtr = &sessionIDHeader
	}
	sessionID, _ := h.sessionStore.GetOrCreate(sessionIDPtr)

	if req.Stream {
		h.agentChatStreaming(w, r, ag, sessionID, req.Message)
		return
	}

	result, err := ag.Chat(r.Context(), sessionID, req.Message, nil)
	if err != nil {
		classifyUpstreamError(err).WriteJSON(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Session-ID", sessionID)
	json.NewEncoder(w).Encode(AgentChatResponse{
		SessionID: sessionID,
		Content:   result.Response,
		Citations: toCitationDTOs(result.Citations),
		Images:    result.Images,
		ToolCalls: result.ToolCalls,
	})
}

// agentChatStreaming streams the agent turn as proprietary JSON-SSE.
func (h *Handler) agentChatStreaming(w http.ResponseWriter, r *http.Request, ag *agent.Agent, sessionID, message string) {
	flusher, err := prepareSSE(w)
	if err != nil {
		ErrInternal(err).WriteJSON(w, r)
		return
	}
	w.Header().Set("X-Session-ID", sessionID)

	sink := &agentSSESink{w: w, flusher: flusher}
	// A RelayChat error here almost always means the client disconnected
	// mid-stream; headers are already flushed so there's nothing to surface to
	// the client — log at debug for observability.
	if err := agent.RelayChat(r.Context(), ag, sessionID, message, sink); err != nil {
		logger.Debug().
			Err(err).
			Str("path", r.URL.Path).
			Str("request_id", middleware.GetReqID(r.Context())).
			Msg("agent chat stream relay ended early (likely client disconnect)")
	}
}

// agentSSESink implements agent.StreamSink for the proprietary API, emitting a
// JSON-SSE vocabulary the admin JS client (and any future client) consumes:
//
//	event: token   data: {"text": "..."}
//	event: tool    data: {"phase":"start"|"done", ...}
//	event: done    data: {"citations":[...],"images":[...],"tool_calls":[...]}
//	event: error   data: {"message": "..."}
type agentSSESink struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func (s *agentSSESink) OnToken(tok string) error {
	return writeSSENamedEvent(s.w, s.flusher, "token", map[string]string{"text": tok})
}

type agentToolEventDTO struct {
	Phase          string           `json:"phase"` // "start" | "done"
	CallID         string           `json:"call_id"`
	Name           string           `json:"name"`
	Args           string           `json:"args,omitempty"`
	DurationMS     int64            `json:"duration_ms,omitempty"`
	Error          bool             `json:"error,omitempty"`
	Citations      []citationDTO    `json:"citations,omitempty"`
	Images         []agent.ImageRef `json:"images,omitempty"`
	VisionAttached bool             `json:"vision_attached,omitempty"`
}

func (s *agentSSESink) OnToolEvent(evt agent.ToolEvent) error {
	switch evt.Type {
	case "start":
		return writeSSENamedEvent(s.w, s.flusher, "tool", agentToolEventDTO{
			Phase:  "start",
			CallID: evt.CallID,
			Name:   evt.Name,
			Args:   evt.Args,
		})
	case "done":
		return writeSSENamedEvent(s.w, s.flusher, "tool", agentToolEventDTO{
			Phase:          "done",
			CallID:         evt.CallID,
			Name:           evt.Name,
			Args:           evt.Args,
			DurationMS:     evt.Duration.Milliseconds(),
			Error:          evt.Error,
			Citations:      toCitationDTOs(evt.Citations),
			Images:         evt.Images,
			VisionAttached: evt.VisionAttached,
		})
	}
	return nil
}

type agentDoneDTO struct {
	Citations []citationDTO          `json:"citations,omitempty"`
	Images    []agent.ImageRef       `json:"images,omitempty"`
	ToolCalls []agent.ToolCallRecord `json:"tool_calls,omitempty"`
}

func (s *agentSSESink) OnEnd(result *agent.ChatResult, agentErr error) error {
	if agentErr != nil {
		return writeSSENamedEvent(s.w, s.flusher, "error", map[string]string{"message": agentErr.Error()})
	}
	done := agentDoneDTO{}
	if result != nil {
		done = agentDoneDTO{
			Citations: toCitationDTOs(result.Citations),
			Images:    result.Images,
			ToolCalls: result.ToolCalls,
		}
	}
	return writeSSENamedEvent(s.w, s.flusher, "done", done)
}
