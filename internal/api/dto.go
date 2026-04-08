// Package api provides HTTP types and handlers.
// The API is OpenAI-compatible where applicable.
package api

import (
	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
)

// ChatCompletionRequest is the OpenAI-compatible chat completion request.
// We use the openai-go types directly for compatibility.
type ChatCompletionRequest = openai.ChatCompletionNewParams

// ChatCompletionResponse is the OpenAI-compatible chat completion response.
type ChatCompletionResponse = openai.ChatCompletion

// ChatCompletionWithSources extends the chat response with source references.
// Used in session mode to return sources collected by the agent's semantic_search tool.
type ChatCompletionWithSources struct {
	openai.ChatCompletion

	// Sources contains the documents used for context.
	Sources []domain.SourceReference `json:"sources,omitempty"`
}

// EmbeddingRequest is the OpenAI-compatible embedding request.
type EmbeddingRequest = openai.EmbeddingNewParams

// EmbeddingResponse is the OpenAI-compatible embedding response.
type EmbeddingResponse = openai.CreateEmbeddingResponse

// IngestRequest is used to add documents to the RAG index.
type IngestRequest struct {
	Documents []DocumentInput `json:"documents"`
}

// DocumentInput represents a document to be ingested.
type DocumentInput struct {
	// ID is the unique document identifier.
	ID string `json:"id"`

	// Title is the document title.
	Title string `json:"title"`

	// Content is the full document text.
	Content string `json:"content"`

	// URL is the optional source URL.
	URL *string `json:"url,omitempty"`

	// Language is "es" or "en".
	Language string `json:"language"`

	// Category classifies the document (degree, course, procedure, staff, news, general).
	Category string `json:"category"`

	// Metadata contains optional key-value pairs.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// IngestResponse is returned after document ingestion.
type IngestResponse struct {
	// IngestedCount is the number of documents successfully ingested.
	IngestedCount int `json:"ingested_count"`

	// ChunksCreated is the total number of chunks created.
	ChunksCreated int `json:"chunks_created"`

	// Errors contains any per-document errors.
	Errors []IngestError `json:"errors,omitempty"`
}

// IngestError describes an error during document ingestion.
type IngestError struct {
	// DocumentID identifies the failed document.
	DocumentID string `json:"document_id"`

	// Error describes what went wrong.
	Error string `json:"error"`
}

// HealthResponse is returned by the health check endpoint.
type HealthResponse struct {
	Status string `json:"status"`
}

// ErrorResponse is the standard error response format.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
