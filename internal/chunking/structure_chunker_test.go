package chunking

import (
	"context"
	"strings"
	"testing"

	"etsi-rag/internal/config"
)

// TestImageMarkersPassthrough verifies that the chunker preserves [[image:p1]]
// markers in chunk content. The markers carry image references for
// vision-capable retrieval and must survive chunking unaltered.
//
// Critically, this test mirrors the parser's dense placement (one marker per
// non-heading content line) so that EVERY chunk produced — not just the
// first one in a section — contains a marker. The naive once-per-page
// placement broke when the chunker sliced sections mid-page; the fix is
// dense per-line marker prefixing in the parser.
func TestImageMarkersPassthrough(t *testing.T) {
	cfg := config.ChunkingConfig{
		MaxChunkSize: 200,
		ChunkOverlap: 0,
		MinChunkSize: 5,
	}
	chunker := NewStructureAwareChunker(cfg, &mockCounter{}, 512)

	content := "## Heading One\n\n" +
		"[[image:p1]] Some content for page 1, line 1.\n" +
		"[[image:p1]] Some content for page 1, line 2.\n\n" +
		"## Heading Two\n\n" +
		"[[image:p2]] Some content for page 2, line 1.\n" +
		"[[image:p2]] Some content for page 2, line 2."

	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}

	// Every chunk that has any non-heading content should contain at least
	// one marker — this is the invariant we need for vision retrieval.
	for i, c := range chunks {
		hasContent := strings.Contains(c.Content, "content for page")
		hasMarker := strings.Contains(c.Content, "[[image:")
		if hasContent && !hasMarker {
			t.Errorf("chunk[%d] has body content but no marker: %q", i, c.Content)
		}
	}
}

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

// TestStructureAwareChunkerDeepBreadcrumbDoesntOversizeChunks is a regression
// for the case where a deeply-nested heading's breadcrumb consumes more tokens
// than max_chunk_size. Previously this produced an effectiveMax of <=0 and the
// oversized-section split sent empty/invalid token slices to the detokenizer,
// which llama-server rejects with 400. The chunker should drop the breadcrumb
// for this section rather than produce an error.
func TestStructureAwareChunkerDeepBreadcrumbDoesntOversizeChunks(t *testing.T) {
	ctx := context.Background()
	// mockCounter counts words as tokens. A long breadcrumb (many words) with a
	// tiny maxTokens reproduces the degenerate effectiveMax scenario.
	deepHeadings := "# One Two Three Four Five\n\n" +
		"## Six Seven Eight Nine Ten\n\n" +
		"### Eleven Twelve Thirteen Fourteen Fifteen\n\n" +
		// Body must be oversized (> chunkSize tokens) to force splitOversizedSection.
		strings.Repeat("word ", 30)

	chunker := NewStructureAwareChunker(config.ChunkingConfig{
		MaxChunkSize: 8, // smaller than the >8-word breadcrumb
		MinChunkSize: 4,
		ChunkOverlap: 2,
		Separators:   []string{" ", ""},
	}, &mockCounter{}, 512)

	chunks, err := chunker.Chunk(ctx, deepHeadings)
	if err != nil {
		t.Fatalf("chunk with deep breadcrumb should succeed, got: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk from oversized section")
	}
	for i, c := range chunks {
		if strings.TrimSpace(c.Content) == "" {
			t.Errorf("chunk %d has empty content", i)
		}
	}
}

// TestHardSplitByTokensGuards verifies the defensive guards in hardSplitByTokens
// that prevent infinite loops and empty-slice detokenize calls when the section
// budget is degenerate.
func TestHardSplitByTokensGuards(t *testing.T) {
	ctx := context.Background()
	chunker := NewStructureAwareChunker(config.ChunkingConfig{
		MaxChunkSize: 10,
		MinChunkSize: 4,
		ChunkOverlap: 2,
	}, &mockCounter{}, 512)

	idx := 0
	// maxTokens <= 0: must return empty without error and without touching the
	// tokenizer (mockCounter would return tokens regardless; we only care the
	// loop doesn't execute).
	chunks, err := chunker.hardSplitByTokens(ctx, "some text here", "Breadcrumb", 0, &idx)
	if err != nil {
		t.Fatalf("maxTokens=0: %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("maxTokens=0: expected no chunks, got %d", len(chunks))
	}

	// empty text → Tokenize returns []; second guard kicks in.
	chunks, err = chunker.hardSplitByTokens(ctx, "", "Breadcrumb", 10, &idx)
	if err != nil {
		t.Fatalf("empty text: %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("empty text: expected no chunks, got %d", len(chunks))
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
