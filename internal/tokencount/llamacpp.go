package tokencount

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"time"
)

// DefaultMaxRetries on 5xx. llama-server can crash
// during heavy ingest; without retries every transient 500 fails a chunker
// /tokenize call and surfaces as a chunking error. Mirrors llm.DefaultMaxRetries.
const DefaultMaxRetries = 10

// LlamaCppCounter uses llama.cpp's /tokenize and /detokenize HTTP endpoints.
type LlamaCppCounter struct {
	baseURL    string
	model      string
	client     *http.Client
	maxRetries int
}

// NewLlamaCppCounter creates a counter that calls llama.cpp tokenization endpoints.
// baseURL should NOT include /v1 (e.g. "http://localhost:8080").
// model is sent in the request body so llama-server knows which model to tokenize with.
func NewLlamaCppCounter(baseURL, model string, client *http.Client) *LlamaCppCounter {
	if client == nil {
		client = http.DefaultClient
	}
	return &LlamaCppCounter{baseURL: baseURL, model: model, client: client, maxRetries: DefaultMaxRetries}
}

// WithMaxRetries overrides the default retry count for 5xx responses. Pass 0
// to disable retries entirely (used in tests so error paths fail fast).
func (l *LlamaCppCounter) WithMaxRetries(n int) *LlamaCppCounter {
	l.maxRetries = n
	return l
}

// post sends payload as JSON to the llama.cpp endpoint op ("tokenize" or
// "detokenize") and returns the status and body of any response below 500.
// Network errors and 5xx responses are retried with exponential backoff,
// since the router may be restarting the upstream model, and fail with
// ErrBackend once the retries run out.
func (l *LlamaCppCounter) post(ctx context.Context, op string, payload any) (int, []byte, error) {
	reqBody, err := json.Marshal(payload)
	if err != nil {
		return 0, nil, fmt.Errorf("marshaling %s request: %w", op, err)
	}

	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.baseURL+"/"+op, bytes.NewReader(reqBody))
		if err != nil {
			return 0, nil, fmt.Errorf("creating %s request: %w", op, err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := l.client.Do(req)
		if err != nil {
			if attempt >= l.maxRetries {
				return 0, nil, fmt.Errorf("%s request: %w: %w", op, ErrBackend, err)
			}
		} else {
			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return 0, nil, fmt.Errorf("%s request: reading response: %w: %w", op, ErrBackend, err)
			}
			if resp.StatusCode < 500 {
				return resp.StatusCode, body, nil
			}
			if attempt >= l.maxRetries {
				return 0, nil, fmt.Errorf("%s request: %w: status %d: %s", op, ErrBackend, resp.StatusCode, snippet(body))
			}
		}

		if !sleepWithCtx(ctx, backoffDelay(attempt)) {
			return 0, nil, fmt.Errorf("%s request: %w", op, ctx.Err())
		}
	}
}

// snippet bounds a response body quoted in an error message.
func snippet(body []byte) string {
	return string(body[:min(len(body), 512)])
}

// backoffDelay returns the wait before attempt n+1 (0-indexed): 200, 400, 800,
// 1600, … ms (doubling) with ±10 % jitter. DefaultMaxRetries (10) adds up to
// ≈ 205 s worst case.
func backoffDelay(attempt int) time.Duration {
	base := 200 * time.Millisecond
	d := base << attempt
	jitter := time.Duration(rand.Int64N(int64(d) / 5))
	return d + jitter - d/10
}

func sleepWithCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
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
	status, respBody, err := l.post(ctx, "tokenize", tokenizeRequest{Content: text, AddSpecial: false, Model: l.model})
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("tokenize: unexpected status %d", status)
	}

	var result tokenizeResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding tokenize response: %w", err)
	}
	return result.Tokens, nil
}

func (l *LlamaCppCounter) Detokenize(ctx context.Context, tokens []int) (string, error) {
	// Short-circuit empty slices — llama-server returns 400 for empty tokens arrays.
	if len(tokens) == 0 {
		return "", nil
	}

	status, respBody, err := l.post(ctx, "detokenize", detokenizeRequest{Tokens: tokens, Model: l.model})
	if err != nil {
		return "", err
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("detokenize: status %d with %d tokens (first=%d last=%d): %s",
			status, len(tokens), tokens[0], tokens[len(tokens)-1], snippet(respBody))
	}

	var result detokenizeResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("decoding detokenize response: %w", err)
	}
	return result.Content, nil
}
