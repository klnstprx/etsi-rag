// Package api provides HTTP types and handlers.
// The API is OpenAI-compatible where applicable.
package api

import (
	"github.com/openai/openai-go"
)

// ChatCompletionRequest is the OpenAI-compatible chat completion request.
// We use the openai-go types directly for compatibility.
type ChatCompletionRequest = openai.ChatCompletionNewParams

// ChatCompletionResponse is the OpenAI-compatible chat completion response.
type ChatCompletionResponse = openai.ChatCompletion

// EmbeddingRequest is the OpenAI-compatible embedding request.
type EmbeddingRequest = openai.EmbeddingNewParams

// EmbeddingResponse is the OpenAI-compatible embedding response.
type EmbeddingResponse = openai.CreateEmbeddingResponse

// RAGChatRequest extends the chat request with RAG-specific fields.
// This is used for our /v1/rag/chat endpoint which adds RAG capabilities.
type RAGChatRequest struct {
	// Messages is the conversation history (OpenAI-compatible).
	Messages []openai.ChatCompletionMessageParamUnion `json:"messages"`

	// Model is the model to use for generation.
	Model string `json:"model"`

	// MaxTokens limits the response length.
	MaxTokens int64 `json:"max_tokens,omitempty"`

	// Temperature controls randomness (0.0-2.0).
	Temperature float64 `json:"temperature,omitempty"`

	// Stream enables streaming responses.
	Stream bool `json:"stream,omitempty"`

	// RAG-specific fields below

	// Language filters RAG context to this language ("es" or "en").
	Language *string `json:"language,omitempty"`

	// TopK limits the number of retrieved chunks.
	TopK *int `json:"top_k,omitempty"`

	// IncludeSources adds source references to the response.
	IncludeSources bool `json:"include_sources,omitempty"`
}

// RAGChatResponse extends the chat response with RAG-specific fields.
type RAGChatResponse struct {
	openai.ChatCompletion

	// Sources contains the documents used for context (if include_sources=true).
	Sources []SourceReference `json:"sources,omitempty"`
}

// SourceReference identifies a source document used in RAG context.
type SourceReference struct {
	// ID is the document identifier.
	ID string `json:"id"`

	// Title is the document title.
	Title string `json:"title"`

	// URL is the optional document URL.
	URL *string `json:"url,omitempty"`

	// ChunkIndex identifies which chunk was used.
	ChunkIndex int `json:"chunk_index"`

	// Score is the similarity score (0.0-1.0).
	Score float32 `json:"score"`
}

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
