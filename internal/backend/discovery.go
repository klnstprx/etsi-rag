// Package backend provides backend discovery: model listing and llama.cpp server probing.
package backend

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// LlamaCppProps holds auto-detected properties from a llama.cpp server.
type LlamaCppProps struct {
	ContextSize  int    // n_ctx from /props
	EmbeddingDim int    // n_embd from /v1/models
	ModelPath    string // model_path from /props
}

// FetchLlamaCppProps queries a llama.cpp server's /props and /v1/models endpoints
// to auto-detect context window size, embedding dimension, and model path.
// baseURL may or may not include /v1 — it will be stripped automatically.
func FetchLlamaCppProps(ctx context.Context, baseURL string, client *http.Client) (*LlamaCppProps, error) {
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
		p, err := fetchProps(ctx, client, baseURL)
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
		ContextSize: propsResult.props.DefaultGenerationSettings.NCtx,
		ModelPath:   propsResult.props.ModelPath,
	}

	modelsResult := <-modelsCh
	if modelsResult.err == nil && len(modelsResult.model.Data) > 0 && modelsResult.model.Data[0].Meta != nil {
		props.EmbeddingDim = modelsResult.model.Data[0].Meta.NEmbd
	}

	return props, nil
}

// FetchModelIDs calls the OpenAI-compatible /models endpoint and returns model IDs.
// Works with both llama.cpp and OpenAI-compatible backends.
// baseURL should include /v1 (e.g. "http://localhost:8080/v1").
func FetchModelIDs(ctx context.Context, baseURL, apiKey string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

	var result modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	ids := make([]string, len(result.Data))
	for i, m := range result.Data {
		ids[i] = m.ID
	}
	return ids, nil
}

type propsResponse struct {
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
	ModelPath string `json:"model_path"`
}

type modelsResponse struct {
	Data []struct {
		ID   string `json:"id"`
		Meta *struct {
			NEmbd int `json:"n_embd"`
		} `json:"meta"`
	} `json:"data"`
}

func fetchProps(ctx context.Context, client *http.Client, baseURL string) (*propsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/props", nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	var result propsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func fetchModelsRaw(ctx context.Context, client *http.Client, baseURL string) (*modelsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	var result modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}
