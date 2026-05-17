package agent

import "sync"

// Handle is the single source of truth for the currently-active Agent.
// rebuildClients constructs a fresh *Agent on every config save (the agent
// is replaced, not mutated); the api and admin HTTP handlers both need to
// pick up that fresh instance without holding stale pointers. Routing both
// through one Handle eliminates the parallel SetAgent fanout the previous
// design required.
//
// A nil agent is a valid state (no [agent] backend configured, or the chat
// client failed to resolve). Callers must nil-check Get() — the existing
// "agent not configured" 503 path covers it.
type Handle struct {
	mu sync.RWMutex
	a  *Agent
}

// NewHandle constructs a handle around an initial agent (may be nil).
func NewHandle(a *Agent) *Handle {
	return &Handle{a: a}
}

// Set replaces the agent. Pass nil to indicate the agent is unavailable.
func (h *Handle) Set(a *Agent) {
	h.mu.Lock()
	h.a = a
	h.mu.Unlock()
}

// Get returns the current agent. May be nil.
func (h *Handle) Get() *Agent {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.a
}
