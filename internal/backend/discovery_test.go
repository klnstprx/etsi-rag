package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchLlamaCppProps(t *testing.T) {
	tests := []struct {
		name           string
		model          string // model param passed to FetchLlamaCppProps
		propsStatus    int
		propsBody      string
		modelsBody     string
		embeddingsBody string // response for /v1/embeddings probe fallback
		wantCtx        int
		wantDim        int
		wantPath       string
		wantErr        bool
	}{
		{
			name:        "both endpoints succeed",
			propsStatus: 200,
			propsBody:   `{"default_generation_settings":{"n_ctx":2048},"model_path":"/models/test.gguf"}`,
			modelsBody:  `{"data":[{"meta":{"n_embd":768}}]}`,
			wantCtx:     2048,
			wantDim:     768,
			wantPath:    "/models/test.gguf",
		},
		{
			name:        "props returns 500",
			propsStatus: 500,
			wantErr:     true,
		},
		{
			name:        "models returns null meta no model param",
			propsStatus: 200,
			propsBody:   `{"default_generation_settings":{"n_ctx":4096},"model_path":"/models/loading.gguf"}`,
			modelsBody:  `{"data":[{"meta":null}]}`,
			wantCtx:     4096,
			wantDim:     0,
			wantPath:    "/models/loading.gguf",
		},
		{
			name:        "models returns empty data",
			propsStatus: 200,
			propsBody:   `{"default_generation_settings":{"n_ctx":512},"model_path":"test.gguf"}`,
			modelsBody:  `{"data":[]}`,
			wantCtx:     512,
			wantDim:     0,
			wantPath:    "test.gguf",
		},
		{
			name:           "router mode falls back to embedding probe",
			model:          "qwen-embeddings",
			propsStatus:    200,
			propsBody:      `{"default_generation_settings":{"n_ctx":8192},"model_path":"/models/embed.gguf"}`,
			modelsBody:     `{"data":[{"id":"qwen-embeddings"}]}`,
			embeddingsBody: `{"data":[{"embedding":[0.1,0.2,0.3,0.4,0.5]}]}`,
			wantCtx:        8192,
			wantDim:        5,
			wantPath:       "/models/embed.gguf",
		},
		{
			name:        "router mode embedding probe fails gracefully",
			model:       "chat-model",
			propsStatus: 200,
			propsBody:   `{"default_generation_settings":{"n_ctx":4096},"model_path":"/models/chat.gguf"}`,
			modelsBody:  `{"data":[{"id":"chat-model"}]}`,
			wantCtx:     4096,
			wantDim:     0,
			wantPath:    "/models/chat.gguf",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mux := http.NewServeMux()
			mux.HandleFunc("GET /props", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.propsStatus)
				if tt.propsBody != "" {
					w.Write([]byte(tt.propsBody))
				}
			})
			mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
				if tt.modelsBody != "" {
					w.Write([]byte(tt.modelsBody))
				} else {
					w.WriteHeader(http.StatusInternalServerError)
				}
			})
			mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
				if tt.embeddingsBody != "" {
					w.Write([]byte(tt.embeddingsBody))
				} else {
					w.WriteHeader(http.StatusBadRequest)
				}
			})

			srv := httptest.NewServer(mux)
			defer srv.Close()

			props, err := FetchLlamaCppProps(context.Background(), srv.URL, tt.model, nil)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if props.ContextSize != tt.wantCtx {
				t.Errorf("ContextSize = %d, want %d", props.ContextSize, tt.wantCtx)
			}
			if props.EmbeddingDim != tt.wantDim {
				t.Errorf("EmbeddingDim = %d, want %d", props.EmbeddingDim, tt.wantDim)
			}
			if props.ModelPath != tt.wantPath {
				t.Errorf("ModelPath = %q, want %q", props.ModelPath, tt.wantPath)
			}
		})
	}
}

// TestFetchLlamaCppProps_VisionDetection covers the SupportsVision flag path.
// The probe trusts only the canonical "modalities.vision" boolean returned
// by /props?model=<id> (verified empirically on the live llama-server
// router build, 2026-04-28).
func TestFetchLlamaCppProps_VisionDetection(t *testing.T) {
	tests := []struct {
		name       string
		propsBody  string
		wantVision bool
		wantAudio  bool
	}{
		{
			name:       "vision enabled (live shape, gemma4-e4b)",
			propsBody:  `{"model_alias":"gemma4-e4b","model_path":"/m.gguf","modalities":{"vision":true,"audio":true},"default_generation_settings":{"n_ctx":32768}}`,
			wantVision: true,
			wantAudio:  true,
		},
		{
			name:       "vision disabled but audio enabled",
			propsBody:  `{"model_alias":"audio-only","model_path":"/m.gguf","modalities":{"vision":false,"audio":true}}`,
			wantVision: false,
			wantAudio:  true,
		},
		{
			name:       "modalities object empty",
			propsBody:  `{"model_alias":"plain","model_path":"/m.gguf","modalities":{}}`,
			wantVision: false,
			wantAudio:  false,
		},
		{
			name:       "modalities field missing entirely (older builds)",
			propsBody:  `{"model_alias":"plain","model_path":"/m.gguf","default_generation_settings":{"n_ctx":2048}}`,
			wantVision: false,
			wantAudio:  false,
		},
		{
			name:       "router-shell response without per-model fields",
			propsBody:  `{"role":"router","model_path":"none","default_generation_settings":{"n_ctx":0}}`,
			wantVision: false,
			wantAudio:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mux := http.NewServeMux()
			mux.HandleFunc("GET /props", func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tt.propsBody))
			})
			mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(`{"data":[]}`))
			})
			srv := httptest.NewServer(mux)
			defer srv.Close()

			props, err := FetchLlamaCppProps(context.Background(), srv.URL, "any", nil)
			if err != nil {
				t.Fatalf("FetchLlamaCppProps: %v", err)
			}
			if props.SupportsVision != tt.wantVision {
				t.Errorf("SupportsVision = %v, want %v", props.SupportsVision, tt.wantVision)
			}
			if props.SupportsAudio != tt.wantAudio {
				t.Errorf("SupportsAudio = %v, want %v", props.SupportsAudio, tt.wantAudio)
			}
		})
	}
}

func TestFetchLlamaCppPropsUnreachable(t *testing.T) {
	_, err := FetchLlamaCppProps(context.Background(), "http://127.0.0.1:1", "", nil)
	if err == nil {
		t.Error("expected error for unreachable server")
	}
}

func TestFetchModelIDs(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[{"id":"model-a"},{"id":"model-b"}]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	ids, err := FetchModelIDs(context.Background(), srv.URL+"/v1", "", nil)
	if err != nil {
		t.Fatalf("FetchModelIDs: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("got %d models, want 2", len(ids))
	}
	if ids[0] != "model-a" || ids[1] != "model-b" {
		t.Errorf("ids = %v, want [model-a model-b]", ids)
	}
}

func TestFetchModelIDsWithAuth(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"data":[{"id":"gpt-4"}]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	ids, err := FetchModelIDs(context.Background(), srv.URL+"/v1", "sk-test", nil)
	if err != nil {
		t.Fatalf("FetchModelIDs: %v", err)
	}
	if len(ids) != 1 || ids[0] != "gpt-4" {
		t.Errorf("ids = %v, want [gpt-4]", ids)
	}
}
