package chunking

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
)

func TestStructureAwareChunker(t *testing.T) {
	ctx := context.Background()
	counter := &mockCounter{}

	tests := []struct {
		name            string
		content         string
		chunkSize       int
		minMaxChunkSize int
		wantMinChunks   int
		wantMaxChunks   int
		wantTitles      []string // expected SectionTitle for each chunk (nil = don't check)
		wantContains    []string // strings that should appear in some chunk
		checkFallback   bool     // true = expects fallback (no headings)
	}{
		{
			name:          "empty content",
			content:       "",
			chunkSize:     100,
			wantMinChunks: 0,
			wantMaxChunks: 0,
		},
		{
			name:          "no headings falls back to token splitting",
			content:       "paragraph one.\n\nparagraph two.\n\nparagraph three.",
			chunkSize:     100,
			wantMinChunks: 1,
			wantMaxChunks: 1,
			checkFallback: true,
		},
		{
			name:          "single section fits in one chunk",
			content:       "## Objetivos\n\nBrief text about objectives.",
			chunkSize:     100,
			wantMinChunks: 1,
			wantMaxChunks: 1,
			wantTitles:    []string{"Objetivos"},
			wantContains:  []string{"## Objetivos", "Brief text"},
		},
		{
			name:          "multiple sections at same level",
			content:       "## Section A\n\nContent for A.\n\n## Section B\n\nContent for B.",
			chunkSize:     100,
			wantMinChunks: 2,
			wantMaxChunks: 2,
			wantTitles:    []string{"Section A", "Section B"},
		},
		{
			name:          "nested headings build breadcrumb",
			content:       "# Title\n\nIntro.\n\n## Sub\n\nSub content.\n\n### Deep\n\nDeep content.",
			chunkSize:     100,
			wantMinChunks: 2,
			wantMaxChunks: 3,
			wantContains:  []string{"Title > Sub > Deep"},
		},
		{
			name:            "large section is split",
			content:         "## Big\n\n" + strings.Repeat("word ", 200),
			chunkSize:       20,
			minMaxChunkSize: 5,
			wantMinChunks:   3,
			wantMaxChunks:   20,
		},
		{
			name:            "small sections merged when below min",
			content:         "## A\n\nShort.\n\n## B\n\nAlso short.",
			chunkSize:       100,
			minMaxChunkSize: 50, // both sections are tiny, should merge
			wantMinChunks:   1,
			wantMaxChunks:   1,
		},
		{
			name:          "table in section kept intact",
			content:       "## Plan\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
			chunkSize:     100,
			wantMinChunks: 1,
			wantMaxChunks: 1,
			wantContains:  []string{"| Col1 | Col2 |"},
		},
		{
			name:          "content before first heading",
			content:       "Some preamble text.\n\n## First Section\n\nSection content.",
			chunkSize:     100,
			wantMinChunks: 2,
			wantMaxChunks: 2,
			wantTitles:    []string{"", "First Section"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config.ChunkingConfig{
				MaxChunkSize: tt.chunkSize,
				ChunkOverlap: 0,
				MinChunkSize: tt.minMaxChunkSize,
			}
			if cfg.MinChunkSize == 0 {
				cfg.MinChunkSize = 5
			}

			chunker := NewStructureAwareChunker(cfg, counter, 512)
			chunks, err := chunker.Chunk(ctx, tt.content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantMinChunks == 0 && tt.wantMaxChunks == 0 {
				if chunks != nil {
					t.Errorf("expected nil for empty content, got %d chunks", len(chunks))
				}
				return
			}

			if len(chunks) < tt.wantMinChunks || len(chunks) > tt.wantMaxChunks {
				t.Errorf("expected %d-%d chunks, got %d", tt.wantMinChunks, tt.wantMaxChunks, len(chunks))
				for i, c := range chunks {
					t.Logf("  chunk[%d] title=%q content=%q", i, c.SectionTitle, truncate(c.Content, 80))
				}
			}

			// Check section titles
			if tt.wantTitles != nil && len(chunks) == len(tt.wantTitles) {
				for i, want := range tt.wantTitles {
					if chunks[i].SectionTitle != want {
						t.Errorf("chunk[%d] SectionTitle = %q, want %q", i, chunks[i].SectionTitle, want)
					}
				}
			}

			// Check content contains
			if tt.wantContains != nil {
				allContent := ""
				for _, c := range chunks {
					allContent += c.Content + "\n"
				}
				for _, want := range tt.wantContains {
					if !strings.Contains(allContent, want) {
						t.Errorf("expected chunk content to contain %q", want)
					}
				}
			}

			// Check fallback: no section titles expected
			if tt.checkFallback {
				for i, c := range chunks {
					if c.SectionTitle != "" {
						t.Errorf("fallback chunk[%d] should have empty SectionTitle, got %q", i, c.SectionTitle)
					}
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

func TestStructureAwareChunkerNilCounter(t *testing.T) {
	ctx := context.Background()
	chunker := NewStructureAwareChunker(config.ChunkingConfig{
		MaxChunkSize: 10,
		MinChunkSize: 5,
	}, nil, 512)

	_, err := chunker.Chunk(ctx, "## Test\n\nsome content here")
	if err == nil {
		t.Error("expected error for nil counter")
	}
}

func TestParseSections(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		wantCount  int
		wantLevels []int
		wantTitles []string
	}{
		{
			name:       "no headings",
			input:      "just text",
			wantCount:  1,
			wantLevels: []int{0},
		},
		{
			name:       "single h2",
			input:      "## Title\n\nBody text.",
			wantCount:  1,
			wantLevels: []int{2},
			wantTitles: []string{"Title"},
		},
		{
			name:       "preamble then heading",
			input:      "Preamble.\n\n## Section\n\nBody.",
			wantCount:  2,
			wantLevels: []int{0, 2},
		},
		{
			name:       "multiple levels",
			input:      "# H1\n\nBody1.\n\n## H2\n\nBody2.\n\n### H3\n\nBody3.",
			wantCount:  3,
			wantLevels: []int{1, 2, 3},
			wantTitles: []string{"H1", "H2", "H3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sections := parseSections(tt.input)
			if len(sections) != tt.wantCount {
				t.Fatalf("expected %d sections, got %d", tt.wantCount, len(sections))
			}
			for i, s := range sections {
				if s.level != tt.wantLevels[i] {
					t.Errorf("section[%d] level = %d, want %d", i, s.level, tt.wantLevels[i])
				}
				if tt.wantTitles != nil && s.title != tt.wantTitles[i] {
					t.Errorf("section[%d] title = %q, want %q", i, s.title, tt.wantTitles[i])
				}
			}
		})
	}
}

func TestBuildBreadcrumbs(t *testing.T) {
	sections := []section{
		{level: 1, title: "Top"},
		{level: 2, title: "Mid"},
		{level: 3, title: "Deep"},
		{level: 2, title: "Mid2"},
		{level: 1, title: "Top2"},
	}

	breadcrumbs := buildBreadcrumbs(sections)

	want := []string{
		"Top",
		"Top > Mid",
		"Top > Mid > Deep",
		"Top > Mid2",
		"Top2",
	}

	for i, w := range want {
		if breadcrumbs[i] != w {
			t.Errorf("breadcrumb[%d] = %q, want %q", i, breadcrumbs[i], w)
		}
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
