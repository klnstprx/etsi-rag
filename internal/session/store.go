// Package session provides conversation session management.
package session

import (
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go"

	"etsi-rag/internal/config"
)

// Session holds conversation state.
type Session struct {
	mu sync.Mutex

	// Messages contains the conversation history (OpenAI-compatible).
	Messages []openai.ChatCompletionMessageParamUnion

	// MaxMessages limits history length.
	MaxMessages int

	// LastAccessed tracks when the session was last used.
	LastAccessed time.Time
}

// AddMessage appends a message, truncating if over limit.
func (s *Session) AddMessage(msg openai.ChatCompletionMessageParamUnion) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, msg)
	if len(s.Messages) > s.MaxMessages {
		s.Messages = s.Messages[1:]
	}
	s.LastAccessed = time.Now()
}

// GetMessages returns a snapshot copy of the message history.
func (s *Session) GetMessages() []openai.ChatCompletionMessageParamUnion {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LastAccessed = time.Now()
	cp := make([]openai.ChatCompletionMessageParamUnion, len(s.Messages))
	copy(cp, s.Messages)
	return cp
}

// Touch updates the last-accessed timestamp.
func (s *Session) Touch() {
	s.mu.Lock()
	s.LastAccessed = time.Now()
	s.mu.Unlock()
}

// Store manages conversation sessions with TTL-based expiration.
type Store struct {
	mu          sync.RWMutex
	sessions    map[string]*Session
	maxMessages int
	ttl         time.Duration
	stopCleanup chan struct{}
}

// NewStore creates a new session store.
// Uses config.SessionConfig from the centralized config package.
func NewStore(cfg config.SessionConfig) *Store {
	s := &Store{
		sessions:    make(map[string]*Session),
		maxMessages: cfg.MaxMessages,
		ttl:         time.Duration(cfg.TTLSeconds) * time.Second,
		stopCleanup: make(chan struct{}),
	}

	go s.cleanupLoop()
	return s
}

// cleanupLoop periodically removes expired sessions.
func (s *Store) cleanupLoop() {
	ticker := time.NewTicker(s.ttl / 2)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.CleanupExpired()
		case <-s.stopCleanup:
			return
		}
	}
}

// Close stops the background cleanup goroutine.
func (s *Store) Close() {
	close(s.stopCleanup)
}

// Create creates a new session and returns its ID.
func (s *Store) Create() string {
	id := uuid.New().String()
	session := &Session{
		Messages:     make([]openai.ChatCompletionMessageParamUnion, 0),
		MaxMessages:  s.maxMessages,
		LastAccessed: time.Now(),
	}

	s.mu.Lock()
	s.sessions[id] = session
	s.mu.Unlock()

	return id
}

// Get returns a session by ID, or nil if not found.
func (s *Store) Get(id string) *Session {
	s.mu.RLock()
	session, exists := s.sessions[id]
	s.mu.RUnlock()

	if !exists {
		return nil
	}

	return session
}

// GetOrCreate returns existing session or creates new one.
func (s *Store) GetOrCreate(id *string) (string, *Session) {
	if id == nil || *id == "" {
		newID := s.Create()
		return newID, s.Get(newID)
	}

	session := s.Get(*id)
	if session == nil {
		newID := s.Create()
		return newID, s.Get(newID)
	}

	session.Touch()
	return *id, session
}

// CleanupExpired removes sessions that have exceeded their TTL.
func (s *Store) CleanupExpired() {
	now := time.Now()

	s.mu.Lock()
	defer s.mu.Unlock()

	for id, session := range s.sessions {
		if now.Sub(session.LastAccessed) > s.ttl {
			delete(s.sessions, id)
		}
	}
}

// Count returns the number of active sessions.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.sessions)
}
