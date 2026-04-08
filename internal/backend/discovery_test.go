package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchLlamaCppProps(t *testing.T) {
	tests := []struct {
		name        string
		propsStatus int
		propsBody   string
		modelsBody  string
		wantCtx     int
		wantDim     int
		wantPath    string
		wantErr     bool
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
			name:        "models returns null meta",
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

			srv := httptest.NewServer(mux)
			defer srv.Close()

			props, err := FetchLlamaCppProps(context.Background(), srv.URL, "", nil)

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

	ids, err := FetchModelIDs(context.Background(), srv.URL+"/v1", "")
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

	ids, err := FetchModelIDs(context.Background(), srv.URL+"/v1", "sk-test")
	if err != nil {
		t.Fatalf("FetchModelIDs: %v", err)
	}
	if len(ids) != 1 || ids[0] != "gpt-4" {
		t.Errorf("ids = %v, want [gpt-4]", ids)
	}
}
