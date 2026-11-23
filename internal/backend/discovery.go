// Package backend provides backend discovery: model listing and llama.cpp server probing.
package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// LlamaCppProps holds auto-detected properties from a llama.cpp server.
//
// All fields are populated from the per-model /props response (i.e.
// /props?model=<id>), which on the router build returns the loaded model's
// own properties — including a structured "modalities" object. We commit to
// that shape rather than scanning speculative field names.
type LlamaCppProps struct {
	// ContextSize is default_generation_settings.n_ctx — already the
	// per-request limit the server enforces (the slot's n_ctx; llama.cpp
	// server-context.cpp sets slot.n_ctx = llama_n_ctx_seq and reports
	// exactly that in /props). Do NOT divide it by TotalSlots: with the
	// unified KV cache the slots share one buffer dynamically, and in split
	// mode the server already reports the per-slot share here. Verified
	// empirically against the router build (n_ctx=32768, total_slots=4):
	// a 9.6k-token prompt is accepted and a 33.6k one rejected at 32768.
	//
	// The value is a property of the running child process, not of the
	// model: with llama-server memory fitting (--fit on) a child launched
	// under VRAM pressure can come up with a smaller context than the same
	// model served later, so callers must treat it as valid-at-probe-time.
	ContextSize    int
	TotalSlots     int    // total_slots (server n_parallel) — informational, logged for concurrency insight
	EmbeddingDim   int    // /v1/models data[].meta.n_embd, with /v1/embeddings probe fallback
	ModelPath      string // model_path
	ModelAlias     string // model_alias — echoes the ?model= query, useful for log correlation
	SupportsVision bool   // modalities.vision
	SupportsAudio  bool   // modalities.audio (recorded for future use; not consumed yet)
}

// FetchLlamaCppProps queries a llama.cpp server's /props and /v1/models endpoints
// to auto-detect context window size, embedding dimension, and model path.
// baseURL may or may not include /v1 — it will be stripped automatically.
// model is passed as a query parameter to /props for model-router support (may be empty).
func FetchLlamaCppProps(ctx context.Context, baseURL, model string, client *http.Client) (*LlamaCppProps, error) {
	baseURL = strings.TrimSuffix(baseURL, "/v1")
	if client == nil {
		client = http.DefaultClient
	}

	// /v1/models only adds the embedding dimension, so it runs alongside
	// /props and its failure is not an error.
	var models modelsResponse
	modelsErr := make(chan error, 1)
	go func() {
		modelsErr <- getJSON(ctx, client, baseURL+"/v1/models", "", &models)
	}()

	propsURL := baseURL + "/props"
	if model != "" {
		propsURL += "?model=" + model
	}
	var p propsResponse
	if err := getJSON(ctx, client, propsURL, "", &p); err != nil {
		return nil, fmt.Errorf("fetching /props: %w", err)
	}

	props := &LlamaCppProps{
		ContextSize:    p.DefaultGenerationSettings.NCtx,
		TotalSlots:     p.TotalSlots,
		ModelPath:      p.ModelPath,
		ModelAlias:     p.ModelAlias,
		SupportsVision: p.Modalities.Vision,
		SupportsAudio:  p.Modalities.Audio,
	}

	if err := <-modelsErr; err == nil && len(models.Data) > 0 && models.Data[0].Meta != nil {
		props.EmbeddingDim = models.Data[0].Meta.NEmbd
	}

	// Fallback: if n_embd is still unknown (e.g. router mode), try a test embedding.
	if props.EmbeddingDim == 0 && model != "" {
		if dim, err := probeEmbeddingDim(ctx, client, baseURL, model); err == nil {
			props.EmbeddingDim = dim
		}
	}

	return props, nil
}

// FetchModelIDs calls the OpenAI-compatible /models endpoint and returns model IDs.
// Works with both llama.cpp and OpenAI-compatible backends.
// baseURL should include /v1 (e.g. "http://localhost:8080/v1").
// client may be nil, in which case http.DefaultClient is used.
func FetchModelIDs(ctx context.Context, baseURL, apiKey string, client *http.Client) ([]string, error) {
	if client == nil {
		client = http.DefaultClient
	}
	var result modelsResponse
	if err := getJSON(ctx, client, baseURL+"/models", apiKey, &result); err != nil {
		return nil, fmt.Errorf("fetching models from %s: %w", baseURL, err)
	}

	ids := make([]string, len(result.Data))
	for i, m := range result.Data {
		ids[i] = m.ID
	}
	return ids, nil
}

// propsResponse mirrors the canonical /props?model=<id> shape returned by
// recent llama-server builds (verified empirically 2026-04-28 against the
// router build). Only the fields the codebase consumes are decoded — the
// response contains plenty more (chat_template, sampler params, etc.) that
// don't belong here.
type propsResponse struct {
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
	// total_slots is a top-level sibling of default_generation_settings, NOT
	// nested inside it.
	TotalSlots int    `json:"total_slots"`
	ModelPath  string `json:"model_path"`
	ModelAlias string `json:"model_alias"`
	Modalities struct {
		Vision bool `json:"vision"`
		Audio  bool `json:"audio"`
	} `json:"modalities"`
}

type modelsResponse struct {
	Data []struct {
		ID   string `json:"id"`
		Meta *struct {
			NEmbd int `json:"n_embd"`
		} `json:"meta"`
	} `json:"data"`
}

// getJSON GETs url and decodes a 200 response body into dst. A non-empty
// apiKey is sent as a bearer token.
func getJSON(ctx context.Context, client *http.Client, url, apiKey string, dst any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("building request: %w", err)
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		return fmt.Errorf("decoding response: %w", err)
	}
	return nil
}

// probeEmbeddingDim sends a short test embedding request and returns the vector dimension.
// Returns 0 and an error if the model doesn't support embeddings or the request fails.
func probeEmbeddingDim(ctx context.Context, client *http.Client, baseURL, model string) (int, error) {
	body, _ := json.Marshal(map[string]string{"input": "dim", "model": model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("building embeddings probe request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("sending embeddings probe: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("embeddings probe: status %d", resp.StatusCode)
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decoding embeddings probe response: %w", err)
	}
	if len(result.Data) == 0 {
		return 0, fmt.Errorf("embeddings probe: empty data")
	}
	return len(result.Data[0].Embedding), nil
}
