package admin

import "sync"

// chatStreamManager holds active admin chat SSE streams, keyed by session ID.
// Writers register a stream when the agent goroutine starts; the SSE handler
// looks it up, consumes its channels, and removes it on disconnect.
type chatStreamManager struct {
	mu      sync.RWMutex
	streams map[string]*chatStream
}

func newChatStreamManager() *chatStreamManager {
	return &chatStreamManager{streams: make(map[string]*chatStream)}
}

func (m *chatStreamManager) register(sessionID string, s *chatStream) {
	m.mu.Lock()
	m.streams[sessionID] = s
	m.mu.Unlock()
}

func (m *chatStreamManager) get(sessionID string) (*chatStream, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	s, ok := m.streams[sessionID]
	return s, ok
}

func (m *chatStreamManager) remove(sessionID string) {
	m.mu.Lock()
	delete(m.streams, sessionID)
	m.mu.Unlock()
}
