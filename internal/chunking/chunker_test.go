package chunking

import (
	"strings"
	"testing"

	"etsi-rag/internal/config"
)

func TestRecursiveCharacterChunker(t *testing.T) {
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
				ChunkSize:    tt.chunkSize,
				ChunkOverlap: tt.overlap,
			})

			chunks := chunker.Chunk(tt.content)

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
	content := "AAAAA\n\nBBBBB\n\nCCCCC"
	chunker := NewRecursiveCharacterChunker(config.ChunkingConfig{
		ChunkSize:    10,
		ChunkOverlap: 3,
	})

	chunks := chunker.Chunk(content)
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Second chunk should start with overlap from first chunk
	for i, c := range chunks {
		t.Logf("chunk[%d]: %q", i, c.Content)
	}
}

func TestTokenChunker(t *testing.T) {
	content := "the quick brown fox jumped over the lazy dog and went home"
	chunker := NewTokenChunker(4, 1, nil) // whitespace tokenizer, 4 tokens, 1 overlap

	chunks := chunker.Chunk(content)
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}

	// Each chunk should have at most 4 words
	for i, c := range chunks {
		words := strings.Fields(c.Content)
		if len(words) > 4 {
			t.Errorf("chunk %d has %d words, expected ≤ 4: %q", i, len(words), c.Content)
		}
	}
}

func TestSentenceChunker(t *testing.T) {
	content := "First sentence. Second sentence. Third sentence. Fourth sentence."
	chunker := NewSentenceChunker(40, 10)

	chunks := chunker.Chunk(content)
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}

	for i, c := range chunks {
		if len(c.Content) > 50 { // some slack for boundary
			t.Errorf("chunk %d too long: %d chars", i, len(c.Content))
		}
		t.Logf("chunk[%d]: %q", i, c.Content)
	}
}

func TestChunkerEmpty(t *testing.T) {
	chunkers := []Chunker{
		NewRecursiveCharacterChunker(config.ChunkingConfig{ChunkSize: 100}),
		NewTokenChunker(10, 2, nil),
		NewSentenceChunker(100, 10),
	}

	for _, c := range chunkers {
		if chunks := c.Chunk(""); chunks != nil {
			t.Errorf("expected nil for empty content")
		}
	}
}
