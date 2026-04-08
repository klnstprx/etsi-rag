package api

import (
	"encoding/json"
	"fmt"
	"net/http"
)

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

// WriteJSON writes the error as JSON to the response.
func (e *AppError) WriteJSON(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Code)
	json.NewEncoder(w).Encode(e)
}

// Error constructors

// ErrConfig creates a configuration error.
func ErrConfig(err error) *AppError {
	return &AppError{
		Code:    http.StatusInternalServerError,
		Message: "Configuration error",
		Err:     err,
	}
}

// ErrBadGateway creates an external service error.
func ErrBadGateway(err error) *AppError {
	return &AppError{
		Code:    http.StatusBadGateway,
		Message: "External service unavailable",
		Err:     err,
	}
}

// ErrDimensionMismatch creates a dimension mismatch error.
func ErrDimensionMismatch(expected, got int) *AppError {
	return &AppError{
		Code:    http.StatusBadRequest,
		Message: fmt.Sprintf("Embedding dimension mismatch: expected %d, got %d", expected, got),
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

// ErrSessionNotFound creates a session not found error.
func ErrSessionNotFound(sessionID string) *AppError {
	return &AppError{
		Code:    http.StatusNotFound,
		Message: fmt.Sprintf("Session not found: %s", sessionID),
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
