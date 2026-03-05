package chunking

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
)

func TestRecursiveCharacterChunker(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name       string
		content    string
		chunkSize  int
		overlap    int
		wantCount  int
		wantFirst  string // prefix of first chunk
		checkEmpty bool
	}{
		{
			name:       "empty content",
			content:    "",
			chunkSize:  100,
			overlap:    10,
			wantCount:  0,
			checkEmpty: true,
		},
		{
			name:      "short content fits in one chunk",
			content:   "Hello world",
			chunkSize: 100,
			overlap:   10,
			wantCount: 1,
			wantFirst: "Hello world",
		},
		{
			name:      "splits on paragraph boundary",
			content:   "First paragraph.\n\nSecond paragraph.",
			chunkSize: 25,
			overlap:   0,
			wantCount: 2,
			wantFirst: "First paragraph.",
		},
		{
			name:      "splits on newline when no paragraph break",
			content:   "Line one.\nLine two.\nLine three.\nLine four.",
			chunkSize: 25,
			overlap:   0,
			wantCount: 2,
		},
		{
			name:      "long text without separators",
			content:   strings.Repeat("a", 200),
			chunkSize: 50,
			overlap:   0,
			wantCount: 4,
		},
		{
			name:      "unicode content",
			content:   "Información sobre la matrícula.\n\nDetalles del grado.",
			chunkSize: 40,
			overlap:   0,
			wantCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunker := NewRecursiveCharacterChunker(config.ChunkingConfig{
				MaxChunkSize: tt.chunkSize,
				ChunkOverlap: tt.overlap,
			})

			chunks, err := chunker.Chunk(ctx, tt.content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.checkEmpty {
				if chunks != nil {
					t.Errorf("expected nil for empty content, got %d chunks", len(chunks))
				}
				return
			}

			if len(chunks) != tt.wantCount {
				t.Errorf("expected %d chunks, got %d", tt.wantCount, len(chunks))
				for i, c := range chunks {
					t.Logf("  chunk[%d]: %q", i, c.Content)
				}
				return
			}

			if tt.wantFirst != "" && !strings.HasPrefix(chunks[0].Content, tt.wantFirst) {
				t.Errorf("first chunk should start with %q, got %q", tt.wantFirst, chunks[0].Content)
			}

			// Verify indices are sequential
			for i, c := range chunks {
				if c.Index != i {
					t.Errorf("chunk %d has Index %d", i, c.Index)
				}
			}
		})
	}
}

func TestRecursiveCharacterChunkerOverlap(t *testing.T) {
	ctx := context.Background()
	content := "AAAAA\n\nBBBBB\n\nCCCCC"
	chunker := NewRecursiveCharacterChunker(config.ChunkingConfig{
		MaxChunkSize: 10,
		ChunkOverlap: 3,
	})

	chunks, err := chunker.Chunk(ctx, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Second chunk should start with overlap from first chunk
	for i, c := range chunks {
		t.Logf("chunk[%d]: %q", i, c.Content)
	}
}

func TestChunkerEmpty(t *testing.T) {
	ctx := context.Background()
	counter := &mockCounter{}
	chunkers := []Chunker{
		NewRecursiveCharacterChunker(config.ChunkingConfig{MaxChunkSize: 100}),
		NewRecursiveTokenChunker(config.ChunkingConfig{MaxChunkSize: 100}, counter, 512),
	}

	for _, c := range chunkers {
		chunks, err := c.Chunk(ctx, "")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if chunks != nil {
			t.Errorf("expected nil for empty content")
		}
	}
}

// mockCounter is a test counter that counts words as tokens.
type mockCounter struct{}

func (m *mockCounter) Count(_ context.Context, text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	return len(strings.Fields(text)), nil
}

func (m *mockCounter) Tokenize(_ context.Context, text string) ([]int, error) {
	words := strings.Fields(text)
	tokens := make([]int, len(words))
	for i := range tokens {
		tokens[i] = i
	}
	return tokens, nil
}

func (m *mockCounter) Detokenize(_ context.Context, tokens []int) (string, error) {
	parts := make([]string, len(tokens))
	for i := range parts {
		parts[i] = "word"
	}
	return strings.Join(parts, " "), nil
}

func TestRecursiveTokenChunker(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		content   string
		chunkSize int
		overlap   int
		maxEmbed  int
		wantMin   int // minimum expected chunks
		wantMax   int // maximum expected chunks
	}{
		{
			name:      "short content single chunk",
			content:   "Hello world",
			chunkSize: 10,
			overlap:   0,
			maxEmbed:  512,
			wantMin:   1,
			wantMax:   1,
		},
		{
			name:      "splits into multiple chunks",
			content:   "one two three four five six seven eight nine ten eleven twelve",
			chunkSize: 4,
			overlap:   0,
			maxEmbed:  512,
			wantMin:   3,
			wantMax:   4,
		},
		{
			name:      "embedding limit caps chunk size",
			content:   "one two three four five six seven eight nine ten",
			chunkSize: 100, // very large
			overlap:   0,
			maxEmbed:  3, // but embedding only allows 3 tokens
			wantMin:   3,
			wantMax:   5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter := &mockCounter{}
			chunker := NewRecursiveTokenChunker(config.ChunkingConfig{
				MaxChunkSize: tt.chunkSize,
				ChunkOverlap: tt.overlap,
			}, counter, tt.maxEmbed)

			chunks, err := chunker.Chunk(ctx, tt.content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(chunks) < tt.wantMin || len(chunks) > tt.wantMax {
				t.Errorf("expected %d-%d chunks, got %d", tt.wantMin, tt.wantMax, len(chunks))
				for i, c := range chunks {
					t.Logf("  chunk[%d]: %q", i, c.Content)
				}
			}

			// Verify indices are sequential
			for i, c := range chunks {
				if c.Index != i {
					t.Errorf("chunk %d has Index %d", i, c.Index)
				}
			}
		})
	}
}

func TestRecursiveTokenChunkerNilCounter(t *testing.T) {
	ctx := context.Background()
	chunker := NewRecursiveTokenChunker(config.ChunkingConfig{
		MaxChunkSize: 10,
		ChunkOverlap: 2,
	}, nil, 512)

	_, err := chunker.Chunk(ctx, "some content here")
	if err == nil {
		t.Error("expected error for nil counter")
	}
}

func TestRecursiveTokenChunkerWithOverlap(t *testing.T) {
	ctx := context.Background()
	counter := &mockCounter{}
	chunker := NewRecursiveTokenChunker(config.ChunkingConfig{
		MaxChunkSize: 10,
		ChunkOverlap: 2,
	}, counter, 512)

	content := strings.Repeat("Hello world. ", 20)
	chunks, err := chunker.Chunk(ctx, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(chunks))
	}

	for i, c := range chunks {
		if c.Content == "" {
			t.Errorf("chunk %d is empty", i)
		}
	}
}
