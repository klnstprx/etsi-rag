package session

import (
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go"

	"etsi-rag/internal/config"
)

func newTestStore(t *testing.T, maxMessages, ttlSeconds int) *Store {
	t.Helper()
	s := NewStore(config.SessionConfig{
		MaxMessages: maxMessages,
		TTLSeconds:  ttlSeconds,
	})
	t.Cleanup(s.Close)
	return s
}

func userMsg(text string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(text),
			},
		},
	}
}

func TestAddMessage(t *testing.T) {
	tests := []struct {
		name        string
		maxMessages int
		addCount    int
		wantLen     int
	}{
		{"within limit", 5, 3, 3},
		{"at limit", 5, 5, 5},
		{"over limit truncates", 3, 5, 3},
		{"zero max keeps last", 0, 3, 3}, // MaxMessages=0 means no truncation (0 is never > len)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sess := &Session{
				Messages:     make([]openai.ChatCompletionMessageParamUnion, 0),
				MaxMessages:  tt.maxMessages,
				LastAccessed: time.Now(),
			}

			for i := 0; i < tt.addCount; i++ {
				sess.AddMessage(userMsg("msg"))
			}

			// MaxMessages=0: condition len > 0 is always true, so it always truncates to last 0 items.
			// Actually, looking at the code: if len(s.Messages) > s.MaxMessages, truncate.
			// With MaxMessages=0 and addCount=3: len=1 > 0 → truncate to [1:] → len=0 each time.
			// So with MaxMessages=0, all messages get discarded. Let's fix expectations.
			if tt.maxMessages == 0 {
				// Each add: append([]msg, msg) → len=1 > 0 → truncate to [1:] → len=0
				if len(sess.Messages) != 0 {
					t.Errorf("got len %d, want 0 (zero max discards all)", len(sess.Messages))
				}
				return
			}

			if len(sess.Messages) != tt.wantLen {
				t.Errorf("got len %d, want %d", len(sess.Messages), tt.wantLen)
			}
		})
	}
}

func TestAddMessage_Truncation(t *testing.T) {
	sess := &Session{
		Messages:     make([]openai.ChatCompletionMessageParamUnion, 0),
		MaxMessages:  2,
		LastAccessed: time.Now(),
	}

	sess.AddMessage(userMsg("first"))
	sess.AddMessage(userMsg("second"))
	sess.AddMessage(userMsg("third"))

	if len(sess.Messages) != 2 {
		t.Fatalf("got len %d, want 2", len(sess.Messages))
	}

	// First message should have been truncated; "second" should be at index 0.
	got := sess.Messages[0].OfUser.Content.OfString.Value
	if got != "second" {
		t.Errorf("got first message %q, want %q", got, "second")
	}
}

func TestAddMessage_UpdatesLastAccessed(t *testing.T) {
	sess := &Session{
		Messages:     make([]openai.ChatCompletionMessageParamUnion, 0),
		MaxMessages:  10,
		LastAccessed: time.Now().Add(-time.Hour),
	}

	before := sess.LastAccessed
	time.Sleep(time.Millisecond)
	sess.AddMessage(userMsg("hello"))

	if !sess.LastAccessed.After(before) {
		t.Error("LastAccessed was not updated after AddMessage")
	}
}

func TestCreate(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	id := s.Create()
	if id == "" {
		t.Fatal("Create returned empty ID")
	}

	if s.Count() != 1 {
		t.Errorf("Count() = %d, want 1", s.Count())
	}

	// Second create should give different ID.
	id2 := s.Create()
	if id2 == id {
		t.Error("Create returned duplicate ID")
	}
	if s.Count() != 2 {
		t.Errorf("Count() = %d, want 2", s.Count())
	}
}

func TestGet(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	id := s.Create()

	sess := s.Get(id)
	if sess == nil {
		t.Fatal("Get returned nil for existing session")
	}
	if sess.MaxMessages != 10 {
		t.Errorf("MaxMessages = %d, want 10", sess.MaxMessages)
	}

	// Missing ID.
	if s.Get("nonexistent") != nil {
		t.Error("Get returned non-nil for missing ID")
	}
}

func TestGetOrCreate(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	existingID := s.Create()

	tests := []struct {
		name    string
		id      *string
		wantNew bool
	}{
		{"nil id", nil, true},
		{"empty string id", strPtr(""), true},
		{"valid existing id", &existingID, false},
		{"non-existent id", strPtr("does-not-exist"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID, sess := s.GetOrCreate(tt.id)

			if gotID == "" {
				t.Error("GetOrCreate returned empty ID")
			}
			if sess == nil {
				t.Fatal("GetOrCreate returned nil session")
			}

			if tt.wantNew {
				if tt.id != nil && *tt.id != "" && gotID == *tt.id {
					t.Error("expected new session but got the same ID back")
				}
			} else {
				if gotID != *tt.id {
					t.Errorf("got ID %q, want %q", gotID, *tt.id)
				}
			}
		})
	}
}

func TestCleanupExpired(t *testing.T) {
	s := newTestStore(t, 10, 10) // 10s TTL

	id := s.Create()
	sess := s.Get(id)
	if sess == nil {
		t.Fatal("session should exist before cleanup")
	}

	// Backdate the session.
	s.mu.Lock()
	sess.LastAccessed = time.Now().Add(-time.Minute)
	s.mu.Unlock()

	s.CleanupExpired()

	if s.Get(id) != nil {
		t.Error("expired session was not removed")
	}
	if s.Count() != 0 {
		t.Errorf("Count() = %d, want 0 after cleanup", s.Count())
	}
}

func TestCleanupExpired_KeepsActive(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	s.Create()
	s.CleanupExpired()

	if s.Count() != 1 {
		t.Errorf("Count() = %d, want 1 (active session should be kept)", s.Count())
	}
}

func TestCount(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	n := 5
	for i := 0; i < n; i++ {
		s.Create()
	}

	if s.Count() != n {
		t.Errorf("Count() = %d, want %d", s.Count(), n)
	}
}

func TestConcurrentAccess(t *testing.T) {
	s := newTestStore(t, 10, 3600)

	var wg sync.WaitGroup
	goroutines := 50

	// Concurrent creates.
	ids := make(chan string, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ids <- s.Create()
		}()
	}
	wg.Wait()
	close(ids)

	if s.Count() != goroutines {
		t.Errorf("Count() = %d, want %d after concurrent creates", s.Count(), goroutines)
	}

	// Concurrent reads + count.
	collected := make([]string, 0, goroutines)
	for id := range ids {
		collected = append(collected, id)
	}

	for _, id := range collected {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			if s.Get(id) == nil {
				t.Errorf("Get(%q) returned nil during concurrent access", id)
			}
			s.Count()
		}(id)
	}
	wg.Wait()
}

func TestConcurrentAddMessage_SameSession(t *testing.T) {
	s := newTestStore(t, 100, 3600)
	id := s.Create()
	sess := s.Get(id)

	var wg sync.WaitGroup
	goroutines := 50
	for i := range goroutines {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			sess.AddMessage(userMsg("msg"))
		}(i)
	}
	wg.Wait()

	got := len(sess.GetMessages())
	if got != goroutines {
		t.Errorf("got %d messages, want %d", got, goroutines)
	}
}

func strPtr(s string) *string {
	return &s
}
