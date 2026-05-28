package api

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/llm"
)

func TestWriteJSON_Logs5xxWithWrappedError(t *testing.T) {
	var buf bytes.Buffer
	prev := logger
	SetLogger(zerolog.New(&buf))
	defer SetLogger(prev)

	cause := errors.New("agent backend exploded")
	appErr := ErrInternal(cause)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	ctx := context.WithValue(req.Context(), middleware.RequestIDKey, "test-request-id")
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	appErr.WriteJSON(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status: want 500, got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "Internal server error") {
		t.Errorf("response body should contain public message, got %q", w.Body.String())
	}
	if strings.Contains(w.Body.String(), "agent backend exploded") {
		t.Errorf("response body must not leak wrapped error, got %q", w.Body.String())
	}

	out := buf.String()
	if !strings.Contains(out, "agent backend exploded") {
		t.Errorf("log should contain wrapped error, got %q", out)
	}
	if !strings.Contains(out, "test-request-id") {
		t.Errorf("log should contain request_id, got %q", out)
	}
	if !strings.Contains(out, "/v1/chat/completions") {
		t.Errorf("log should contain path, got %q", out)
	}
}

func TestWriteJSON_DoesNotLog4xx(t *testing.T) {
	var buf bytes.Buffer
	prev := logger
	SetLogger(zerolog.New(&buf))
	defer SetLogger(prev)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	ErrBadRequest("missing field foo").WriteJSON(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status: want 400, got %d", w.Code)
	}
	if buf.Len() != 0 {
		t.Errorf("4xx errors should not be logged, got %q", buf.String())
	}
}

func TestWriteJSON_Skips5xxWithoutWrappedError(t *testing.T) {
	var buf bytes.Buffer
	prev := logger
	SetLogger(zerolog.New(&buf))
	defer SetLogger(prev)

	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	w := httptest.NewRecorder()

	ErrServiceUnavailable("agent not configured").WriteJSON(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: want 503, got %d", w.Code)
	}
	if buf.Len() != 0 {
		t.Errorf("AppError without Err should not log, got %q", buf.String())
	}
}

func TestClassifyUpstreamError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantCode int
		wantMsg  string
	}{
		{
			name:     "empty completion -> 502",
			err:      fmt.Errorf("agent chat: %w", llm.ErrEmptyCompletion),
			wantCode: http.StatusBadGateway,
			wantMsg:  "Empty response from LLM",
		},
		{
			name:     "empty embedding -> 502",
			err:      fmt.Errorf("embedding query: %w", llm.ErrEmptyEmbedding),
			wantCode: http.StatusBadGateway,
			wantMsg:  "Empty response from embedding service",
		},
		{
			name:     "upstream HTTP error -> 502 bad gateway",
			err:      fmt.Errorf("chat with tools: %w", &openai.Error{StatusCode: http.StatusTooManyRequests}),
			wantCode: http.StatusBadGateway,
			wantMsg:  "External service unavailable",
		},
		{
			name:     "generic error -> 500",
			err:      errors.New("something internal broke"),
			wantCode: http.StatusInternalServerError,
			wantMsg:  "Internal server error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyUpstreamError(tt.err)
			if got.Code != tt.wantCode {
				t.Errorf("code = %d, want %d", got.Code, tt.wantCode)
			}
			if got.Message != tt.wantMsg {
				t.Errorf("message = %q, want %q", got.Message, tt.wantMsg)
			}
		})
	}
}
