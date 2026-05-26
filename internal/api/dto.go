// Package api provides HTTP types and handlers.
// The API is OpenAI-compatible where applicable.
package api

import (
	"encoding/json"

	"github.com/openai/openai-go"
)

// ChatCompletionRequest is the OpenAI-compatible chat completion request.
// We embed openai-go's params for field-shape compatibility and add the
// `stream` flag here because the SDK splits New / NewStreaming at the call
// site instead of exposing a Stream field on the params struct.
//
// The embedded type has its own MarshalJSON/UnmarshalJSON; we have to override
// both on the wrapper or the Stream field gets dropped on the wire.
type ChatCompletionRequest struct {
	openai.ChatCompletionNewParams
	Stream bool `json:"stream,omitempty"`
}

// MarshalJSON delegates to the embedded SDK marshaler, then re-inserts the
// stream field at the top level.
func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	inner, err := json.Marshal(r.ChatCompletionNewParams)
	if err != nil {
		return nil, err
	}
	if !r.Stream {
		return inner, nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(inner, &m); err != nil {
		return nil, err
	}
	m["stream"] = json.RawMessage("true")
	return json.Marshal(m)
}

// UnmarshalJSON delegates to the embedded SDK unmarshaler for OpenAI fields,
// then independently decodes the wrapper-only Stream field.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &r.ChatCompletionNewParams); err != nil {
		return err
	}
	var aux struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	r.Stream = aux.Stream
	return nil
}

// ChatCompletionResponse is the OpenAI-compatible chat completion response.
type ChatCompletionResponse = openai.ChatCompletion

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

// RAGQueryRequest is used to inspect the retrieval layer directly, without
// going through the agent's tool-calling loop. Primary consumer: the eval
// harness, which uses the response to compute Recall@k / MRR against a
// ground-truth set of golden URLs.
type RAGQueryRequest struct {
	// Query is the natural-language query to retrieve against.
	Query string `json:"query"`

	// TopK overrides the configured top_k for this request only. Optional.
	TopK *int `json:"top_k,omitempty"`
}

// RAGQueryResponse is the ranked chunk list returned by /v1/rag/query.
type RAGQueryResponse struct {
	Results []RetrievedChunk `json:"results"`
}

// RetrievedChunk is one chunk returned by the retrieval layer, in the shape
// the eval harness needs to map results back to ground truth.
type RetrievedChunk struct {
	ChunkID       int64   `json:"chunk_id"`
	DocumentID    string  `json:"document_id"`
	DocumentTitle string  `json:"document_title"`
	DocumentURL   *string `json:"document_url,omitempty"`
	SectionTitle  string  `json:"section_title,omitempty"`
	ChunkIndex    int     `json:"chunk_index"`
	Score         float32 `json:"score"`
	Snippet       string  `json:"snippet"`
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
