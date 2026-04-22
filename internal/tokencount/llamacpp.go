package tokencount

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// LlamaCppCounter uses llama.cpp's /tokenize and /detokenize HTTP endpoints.
type LlamaCppCounter struct {
	baseURL string
	model   string
	client  *http.Client
}

// NewLlamaCppCounter creates a counter that calls llama.cpp tokenization endpoints.
// baseURL should NOT include /v1 (e.g. "http://localhost:8080").
// model is sent in the request body so llama-server knows which model to tokenize with.
func NewLlamaCppCounter(baseURL, model string, client *http.Client) *LlamaCppCounter {
	if client == nil {
		client = http.DefaultClient
	}
	return &LlamaCppCounter{baseURL: baseURL, model: model, client: client}
}

type tokenizeRequest struct {
	Content    string `json:"content"`
	AddSpecial bool   `json:"add_special"`
	Model      string `json:"model,omitempty"`
}

type tokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

type detokenizeRequest struct {
	Tokens []int  `json:"tokens"`
	Model  string `json:"model,omitempty"`
}

type detokenizeResponse struct {
	Content string `json:"content"`
}

func (l *LlamaCppCounter) Count(ctx context.Context, text string) (int, error) {
	tokens, err := l.Tokenize(ctx, text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

func (l *LlamaCppCounter) Tokenize(ctx context.Context, text string) ([]int, error) {
	body, err := json.Marshal(tokenizeRequest{Content: text, AddSpecial: false, Model: l.model})
	if err != nil {
		return nil, fmt.Errorf("marshaling tokenize request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.baseURL+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating tokenize request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tokenize request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tokenize: unexpected status %d", resp.StatusCode)
	}

	var result tokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding tokenize response: %w", err)
	}
	return result.Tokens, nil
}

func (l *LlamaCppCounter) Detokenize(ctx context.Context, tokens []int) (string, error) {
	// Short-circuit empty slices — llama-server returns 400 for empty tokens arrays.
	if len(tokens) == 0 {
		return "", nil
	}
	body, err := json.Marshal(detokenizeRequest{Tokens: tokens, Model: l.model})
	if err != nil {
		return "", fmt.Errorf("marshaling detokenize request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.baseURL+"/detokenize", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("creating detokenize request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("detokenize request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("detokenize: status %d with %d tokens (first=%d last=%d): %s",
			resp.StatusCode, len(tokens), tokens[0], tokens[len(tokens)-1], string(body))
	}

	var result detokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decoding detokenize response: %w", err)
	}
	return result.Content, nil
}
