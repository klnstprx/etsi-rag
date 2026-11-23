package api

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go"
	"github.com/rs/zerolog"

	"etsi-rag/internal/llm"
)

// logger is the package-level logger used by AppError.WriteJSON to surface
// server-side error causes (the wrapped Err that's stripped from the client
// response). Defaults to a no-op so tests that don't care about logs work
// without setup; main wires the real logger via SetLogger.
var logger = zerolog.Nop()

// SetLogger configures the package-level logger. Call once during server setup.
func SetLogger(l zerolog.Logger) {
	logger = l
}

// AppError represents an application error with an HTTP status code.
type AppError struct {
	Code    int    `json:"-"`
	Message string `json:"error"`
	Err     error  `json:"-"`
}

// Error implements the error interface.
func (e *AppError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Err)
	}
	return e.Message
}

// Unwrap returns the underlying error.
func (e *AppError) Unwrap() error {
	return e.Err
}

// WriteJSON writes the error as JSON to the response. For 5xx errors with a
// wrapped cause, also emits a log line — Err is json:"-" so the client never
// sees it, and without this hook the underlying failure would be invisible.
func (e *AppError) WriteJSON(w http.ResponseWriter, r *http.Request) {
	if e.Code >= 500 && e.Err != nil {
		logger.Error().
			Err(e.Err).
			Int("status", e.Code).
			Str("method", r.Method).
			Str("path", r.URL.Path).
			Str("request_id", middleware.GetReqID(r.Context())).
			Msg("request error")
	}
	writeJSON(w, e.Code, e)
}

// Error constructors

// ErrBadGateway creates an external service error.
func ErrBadGateway(err error) *AppError {
	return &AppError{
		Code:    http.StatusBadGateway,
		Message: "External service unavailable",
		Err:     err,
	}
}

// ErrEmptyLLMResponse creates an empty LLM response error.
func ErrEmptyLLMResponse() *AppError {
	return &AppError{
		Code:    http.StatusBadGateway,
		Message: "Empty response from LLM",
	}
}

// ErrEmptyEmbeddingResponse creates an empty embedding response error.
func ErrEmptyEmbeddingResponse() *AppError {
	return &AppError{
		Code:    http.StatusBadGateway,
		Message: "Empty response from embedding service",
	}
}

// ErrInvalidLanguage creates an invalid language error.
func ErrInvalidLanguage(lang string) *AppError {
	return &AppError{
		Code:    http.StatusBadRequest,
		Message: fmt.Sprintf("Invalid language: %s", lang),
	}
}

// ErrBadRequest creates a bad request error.
func ErrBadRequest(msg string) *AppError {
	return &AppError{
		Code:    http.StatusBadRequest,
		Message: msg,
	}
}

// ErrInternal creates an internal server error.
func ErrInternal(err error) *AppError {
	return &AppError{
		Code:    http.StatusInternalServerError,
		Message: "Internal server error",
		Err:     err,
	}
}

// ErrNotFound creates a not found error.
func ErrNotFound(msg string) *AppError {
	return &AppError{
		Code:    http.StatusNotFound,
		Message: msg,
	}
}

// ErrServiceUnavailable creates a 503 service unavailable error.
func ErrServiceUnavailable(msg string) *AppError {
	return &AppError{
		Code:    http.StatusServiceUnavailable,
		Message: msg,
	}
}

// classifyUpstreamError maps an error from the agent / retrieval / embedding
// path to the right HTTP status. Failures that originate in the model server —
// an empty response, or any upstream HTTP error — are 502s, not 500s: the
// fault is the gateway's, not ours. Everything else stays a 500. Sentinels are
// matched through %w wrapping via errors.Is/As, so callers can wrap freely.
func classifyUpstreamError(err error) *AppError {
	switch {
	case errors.Is(err, llm.ErrEmptyCompletion):
		return ErrEmptyLLMResponse()
	case errors.Is(err, llm.ErrEmptyEmbedding):
		return ErrEmptyEmbeddingResponse()
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return ErrBadGateway(err)
	}
	return ErrInternal(err)
}
