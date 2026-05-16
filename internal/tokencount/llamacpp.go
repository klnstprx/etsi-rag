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

// doWithRetry performs the HTTP call and retries on 5xx with exponential
// backoff. Non-5xx errors return immediately. The returned response body has
// been read into memory and the underlying response closed.
func (l *LlamaCppCounter) doWithRetry(req *http.Request) (status int, body []byte, err error) {
	// Body must be replayable for retries; require requests to use bytes.Reader.
	var rawBody []byte
	if req.Body != nil {
		rawBody, err = io.ReadAll(req.Body)
		if err != nil {
			return 0, nil, fmt.Errorf("buffering request body: %w", err)
		}
		req.Body.Close()
	}

	for attempt := 0; ; attempt++ {
		if rawBody != nil {
			req.Body = io.NopCloser(bytes.NewReader(rawBody))
		}
		resp, doErr := l.client.Do(req)
		if doErr != nil {
			// Network error — retry too, since router may be restarting upstream.
			if attempt < l.maxRetries {
				if !sleepWithCtx(req.Context(), backoffDelay(attempt)) {
					return 0, nil, req.Context().Err()
				}
				continue
			}
			return 0, nil, doErr
		}
		body, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return resp.StatusCode, nil, fmt.Errorf("reading response: %w", err)
		}
		if resp.StatusCode < 500 || attempt >= l.maxRetries {
			return resp.StatusCode, body, nil
		}
		if !sleepWithCtx(req.Context(), backoffDelay(attempt)) {
			return 0, nil, req.Context().Err()
		}
	}
}

// backoffDelay returns the wait before attempt n+1 (0-indexed): 200, 400, 800,
// 1600, 3200 ms with ±20 % jitter. Five retries total ≈ 6.2 s worst case.
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
	body, err := json.Marshal(tokenizeRequest{Content: text, AddSpecial: false, Model: l.model})
	if err != nil {
		return nil, fmt.Errorf("marshaling tokenize request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.baseURL+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating tokenize request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	status, respBody, err := l.doWithRetry(req)
	if err != nil {
		return nil, fmt.Errorf("tokenize request: %w", err)
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
	body, err := json.Marshal(detokenizeRequest{Tokens: tokens, Model: l.model})
	if err != nil {
		return "", fmt.Errorf("marshaling detokenize request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.baseURL+"/detokenize", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("creating detokenize request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	status, respBody, err := l.doWithRetry(req)
	if err != nil {
		return "", fmt.Errorf("detokenize request: %w", err)
	}
	if status != http.StatusOK {
		snippet := respBody
		if len(snippet) > 512 {
			snippet = snippet[:512]
		}
		return "", fmt.Errorf("detokenize: status %d with %d tokens (first=%d last=%d): %s",
			status, len(tokens), tokens[0], tokens[len(tokens)-1], string(snippet))
	}

	var result detokenizeResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("decoding detokenize response: %w", err)
	}
	return result.Content, nil
}
