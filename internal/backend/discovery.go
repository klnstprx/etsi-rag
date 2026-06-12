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

	type result struct {
		props *propsResponse
		model *modelsResponse
		err   error
	}

	propsCh := make(chan result, 1)
	modelsCh := make(chan result, 1)

	go func() {
		p, err := fetchProps(ctx, client, baseURL, model)
		propsCh <- result{props: p, err: err}
	}()
	go func() {
		m, err := fetchModelsRaw(ctx, client, baseURL+"/v1")
		modelsCh <- result{model: m, err: err}
	}()

	propsResult := <-propsCh
	if propsResult.err != nil {
		return nil, fmt.Errorf("fetching /props: %w", propsResult.err)
	}

	props := &LlamaCppProps{
		ContextSize:    propsResult.props.DefaultGenerationSettings.NCtx,
		TotalSlots:     propsResult.props.TotalSlots,
		ModelPath:      propsResult.props.ModelPath,
		ModelAlias:     propsResult.props.ModelAlias,
		SupportsVision: propsResult.props.Modalities.Vision,
		SupportsAudio:  propsResult.props.Modalities.Audio,
	}

	modelsResult := <-modelsCh
	if modelsResult.err == nil && len(modelsResult.model.Data) > 0 && modelsResult.model.Data[0].Meta != nil {
		props.EmbeddingDim = modelsResult.model.Data[0].Meta.NEmbd
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
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("building models request: %w", err)
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching models from %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

	var result modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding models response: %w", err)
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

func fetchProps(ctx context.Context, client *http.Client, baseURL, model string) (*propsResponse, error) {
	u := baseURL + "/props"
	if model != "" {
		u += "?model=" + model
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, fmt.Errorf("building props request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching /props: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	var result propsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding /props response: %w", err)
	}
	return &result, nil
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

func fetchModelsRaw(ctx context.Context, client *http.Client, baseURL string) (*modelsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("building models request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching /models: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	var result modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding /models response: %w", err)
	}
	return &result, nil
}
