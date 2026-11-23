package prompts

import (
	"testing"
)

func TestRender(t *testing.T) {
	tests := []struct {
		name    string
		tmpl    string
		data    Data
		want    string
		wantErr bool
	}{
		{
			name: "plain text without variables",
			tmpl: "You are a helpful assistant.",
			data: Data{DocumentCount: 10, ChunkCount: 50},
			want: "You are a helpful assistant.",
		},
		{
			name: "template with DocumentCount and ChunkCount",
			tmpl: "KB has {{.DocumentCount}} docs and {{.ChunkCount}} chunks.",
			data: Data{DocumentCount: 42, ChunkCount: 256},
			want: "KB has 42 docs and 256 chunks.",
		},
		{
			name: "zero values render as 0",
			tmpl: "Docs: {{.DocumentCount}}, Chunks: {{.ChunkCount}}",
			data: Data{},
			want: "Docs: 0, Chunks: 0",
		},
		{
			name: "empty template string",
			tmpl: "",
			data: Data{DocumentCount: 5, ChunkCount: 20},
			want: "",
		},
		{
			name:    "invalid template syntax",
			tmpl:    "broken {{.Unclosed",
			data:    Data{},
			wantErr: true,
		},
		{
			name: "default Agent template renders without error",
			tmpl: Agent,
			data: Data{DocumentCount: 100, ChunkCount: 500},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := render(tt.tmpl, tt.data)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.want != "" && got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}
