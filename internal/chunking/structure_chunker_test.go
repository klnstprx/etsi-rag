package chunking

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"etsi-rag/internal/config"
)

// mockCounter counts words as tokens. Kept local to the chunking package
// because testutil.MockCounter (functionally identical) lives in
// internal/testutil — which imports chunking — and package-internal tests
// importing testutil would create an import cycle.
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

// TestImageMarkers verifies the parser→chunker handoff: page markers in the
// input drive each chunk's ImageRefs, and the markers themselves are stripped
// from the user-facing chunk content. The parser emits one marker per page
// top; the chunker stamps each emitted chunk with the union of refs whose
// page range overlaps the chunk's source section.
func TestImageMarkers(t *testing.T) {
	cfg := config.ChunkingConfig{
		MaxChunkSize: 200,
		ChunkOverlap: 0,
		MinChunkSize: 5,
	}
	chunker := NewStructureAwareChunker(cfg, &mockCounter{}, 512)

	content := "[[image:p1]]\n" +
		"## Heading One\n\n" +
		"Some content for page 1.\n\n" +
		"[[image:p2]]\n" +
		"## Heading Two\n\n" +
		"Some content for page 2."

	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	for i, c := range chunks {
		if strings.Contains(c.Content, "[[image:") {
			t.Errorf("chunk[%d] still contains a marker: %q", i, c.Content)
		}
	}

	if got := chunks[0].ImageRefs; !reflect.DeepEqual(got, []string{"p1"}) {
		t.Errorf("chunk[0].ImageRefs = %v, want [p1]", got)
	}
	if got := chunks[1].ImageRefs; !reflect.DeepEqual(got, []string{"p2"}) {
		t.Errorf("chunk[1].ImageRefs = %v, want [p2]", got)
	}
}

// TestImageMarkersSpanningSection covers a section whose body straddles a
// page boundary: every emitted chunk should carry both refs.
func TestImageMarkersSpanningSection(t *testing.T) {
	cfg := config.ChunkingConfig{
		MaxChunkSize: 200,
		ChunkOverlap: 0,
		MinChunkSize: 5,
	}
	chunker := NewStructureAwareChunker(cfg, &mockCounter{}, 512)

	content := "[[image:p1]]\n" +
		"## Long Section\n\n" +
		"First half on page 1.\n\n" +
		"[[image:p2]]\n" +
		"Second half on page 2."

	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk (section fits), got %d", len(chunks))
	}
	if got := chunks[0].ImageRefs; !reflect.DeepEqual(got, []string{"p1", "p2"}) {
		t.Errorf("chunk[0].ImageRefs = %v, want [p1 p2]", got)
	}
	if strings.Contains(chunks[0].Content, "[[image:") {
		t.Errorf("chunk content should be marker-free: %q", chunks[0].Content)
	}
}

// TestImageMarkersFallback covers the no-headings path: every emitted chunk
// gets the union of all refs in the input.
func TestImageMarkersFallback(t *testing.T) {
	cfg := config.ChunkingConfig{
		MaxChunkSize: 200,
		ChunkOverlap: 0,
		MinChunkSize: 5,
	}
	chunker := NewStructureAwareChunker(cfg, &mockCounter{}, 512)

	content := "[[image:full]]\nSome plain content with no headings at all."
	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	for i, c := range chunks {
		if !reflect.DeepEqual(c.ImageRefs, []string{"full"}) {
			t.Errorf("chunk[%d].ImageRefs = %v, want [full]", i, c.ImageRefs)
		}
		if strings.Contains(c.Content, "[[image:") {
			t.Errorf("chunk[%d] still contains a marker: %q", i, c.Content)
		}
	}
}

// TestImageMarkersOversizedSplit covers a section that exceeds maxTokens and
// gets split into multiple chunks: each sub-chunk inherits the section refs.
func TestImageMarkersOversizedSplit(t *testing.T) {
	cfg := config.ChunkingConfig{
		MaxChunkSize: 8,
		ChunkOverlap: 0,
		MinChunkSize: 4,
		Separators:   []string{" ", ""},
	}
	chunker := NewStructureAwareChunker(cfg, &mockCounter{}, 512)

	content := "[[image:p1]]\n" +
		"## Big\n\n" +
		strings.Repeat("word ", 30) +
		"\n\n[[image:p2]]\n" +
		strings.Repeat("blob ", 30)

	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected oversized section to split, got %d chunks", len(chunks))
	}
	for i, c := range chunks {
		if !reflect.DeepEqual(c.ImageRefs, []string{"p1", "p2"}) {
			t.Errorf("chunk[%d].ImageRefs = %v, want [p1 p2]", i, c.ImageRefs)
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
				Separators:   []string{"\n\n", "\n", ". ", " ", ""}, // config.default.toml
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

			// Check content contains. SectionTitle is included in the
			// scan so assertions on breadcrumb text still match — the
			// chunker no longer inlines the breadcrumb into Content.
			if tt.wantContains != nil {
				allContent := ""
				for _, c := range chunks {
					allContent += c.SectionTitle + "\n" + c.Content + "\n"
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
// that prevent infinite loops and empty-slice detokenize calls when the budget
// is degenerate.
func TestHardSplitByTokensGuards(t *testing.T) {
	ctx := context.Background()
	counter := &mockCounter{}

	// maxTokens <= 0: must return empty without error and without touching the
	// tokenizer (mockCounter would return tokens regardless; we only care the
	// loop doesn't execute).
	pieces, err := hardSplitByTokens(ctx, counter, "some text here", 0, 2)
	if err != nil {
		t.Fatalf("maxTokens=0: %v", err)
	}
	if len(pieces) != 0 {
		t.Errorf("maxTokens=0: expected no pieces, got %d", len(pieces))
	}

	// empty text → Tokenize returns []; second guard kicks in.
	pieces, err = hardSplitByTokens(ctx, counter, "", 10, 2)
	if err != nil {
		t.Fatalf("empty text: %v", err)
	}
	if len(pieces) != 0 {
		t.Errorf("empty text: expected no pieces, got %d", len(pieces))
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

// TestStructureAwareChunkerKeepsBodilessHeadings: a heading with no body is
// only redundant when a child section follows, because the child's breadcrumb
// carries it. Otherwise its text must still reach a chunk.
func TestStructureAwareChunkerKeepsBodilessHeadings(t *testing.T) {
	type out struct{ title, content string }
	tests := []struct {
		name     string
		content  string
		minChunk int
		want     []out
	}{
		{
			name:     "bodiless heading followed by a sibling",
			content:  "# Centro\n\nIntro.\n\n## Contacto\n\n## Horario\n\nDe nueve a dos.",
			minChunk: 1,
			want: []out{
				{"Centro", "# Centro\n\nIntro."},
				{"Centro > Contacto", "## Contacto"},
				{"Centro > Horario", "## Horario\n\nDe nueve a dos."},
			},
		},
		{
			name:     "bodiless heading followed by a parent-level heading",
			content:  "# A\n\nIntro.\n\n## Enlaces\n\n# B\n\nBody.",
			minChunk: 1,
			want: []out{
				{"A", "# A\n\nIntro."},
				{"A > Enlaces", "## Enlaces"},
				{"B", "# B\n\nBody."},
			},
		},
		{
			name:     "bodiless parent is carried by its child's breadcrumb",
			content:  "# Grado\n\n## Plan\n\nAsignaturas.",
			minChunk: 1,
			want: []out{
				{"Grado > Plan", "## Plan\n\nAsignaturas."},
			},
		},
		{
			name:     "page made only of headings",
			content:  "## Inicio\n\n## Noticias\n\n## Contacto",
			minChunk: 64,
			want: []out{
				{"", "## Inicio\n\n## Noticias"},
				{"Contacto", "## Contacto"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunker := NewStructureAwareChunker(config.ChunkingConfig{
				MaxChunkSize: 100,
				MinChunkSize: tt.minChunk,
				Separators:   []string{"\n\n", "\n", ". ", " ", ""},
			}, &mockCounter{}, 512)
			chunks, err := chunker.Chunk(context.Background(), tt.content)
			if err != nil {
				t.Fatalf("Chunk: %v", err)
			}
			got := make([]out, len(chunks))
			for i, c := range chunks {
				got[i] = out{c.SectionTitle, c.Content}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("chunks = %q\nwant     %q", got, tt.want)
			}
		})
	}
}

// TestStructureAwareChunkerSplitsOversizedTableByRows: a table too large for
// one chunk is a single "\n\n" segment. It must be split on the finer "\n"
// separator so each chunk holds whole rows, not cut by token count.
func TestStructureAwareChunkerSplitsOversizedTableByRows(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("## Horario\n\nClases del primer curso.\n\n| Asignatura | Día | Aula |\n| --- | --- | --- |\n")
	for i := range 30 {
		fmt.Fprintf(&sb, "| Programación %d | Lunes | Aula %d |\n", i, i)
	}
	content := sb.String()

	chunker := NewStructureAwareChunker(config.ChunkingConfig{
		MaxChunkSize: 60,
		MinChunkSize: 5,
		Separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}, &mockCounter{}, 512)
	chunks, err := chunker.Chunk(context.Background(), content)
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected the table to span several chunks, got %d", len(chunks))
	}

	inputLines := strings.Split(content, "\n")
	var rows int
	for i, c := range chunks {
		for line := range strings.SplitSeq(c.Content, "\n") {
			if !slices.Contains(inputLines, line) {
				t.Errorf("chunk[%d] has a line that is not a whole input line: %q", i, line)
			}
			if strings.HasPrefix(line, "| Programación") {
				rows++
			}
		}
	}
	if rows != 30 {
		t.Errorf("chunks hold %d table rows, want all 30", rows)
	}
}

// TestStructureAwareChunkerOverlapsCutParagraphs: when a paragraph or table
// is too large for one chunk and gets cut between sentences or rows, each
// chunk after the first repeats the trailing sentences or rows of the
// previous one, up to chunk_overlap tokens. Chunks that break between whole
// paragraphs stand alone.
func TestStructureAwareChunkerOverlapsCutParagraphs(t *testing.T) {
	tests := []struct {
		name    string
		content string
		max     int
		want    []string
	}{
		{
			name: "long paragraph cut between sentences",
			content: "## Normativa\n\nLa regla uno aplica. La regla dos aplica. " +
				"La regla tres aplica. La regla cuatro aplica. La regla cinco aplica.",
			max: 13,
			want: []string{
				"## Normativa",
				"La regla uno aplica. La regla dos aplica. La regla tres aplica.",
				"La regla tres aplica. La regla cuatro aplica. La regla cinco aplica.",
			},
		},
		{
			name:    "table cut between rows",
			content: "## Horario\n\n| Lunes | 1 |\n| Martes | 2 |\n| Jueves | 3 |\n| Viernes | 4 |",
			max:     13,
			want: []string{
				"## Horario",
				"| Lunes | 1 |\n| Martes | 2 |",
				"| Martes | 2 |\n| Jueves | 3 |",
				"| Jueves | 3 |\n| Viernes | 4 |",
			},
		},
		{
			name:    "chunks split between paragraphs do not overlap",
			content: "## Avisos\n\nPrimer aviso corto.\n\nSegundo aviso corto.\n\nTercer aviso corto.",
			max:     8,
			want: []string{
				"## Avisos\n\nPrimer aviso corto.",
				"Segundo aviso corto.\n\nTercer aviso corto.",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunker := NewStructureAwareChunker(config.ChunkingConfig{
				MaxChunkSize: tt.max,
				MinChunkSize: 1,
				ChunkOverlap: 5,
				Separators:   []string{"\n\n", "\n", ". ", " ", ""},
			}, &mockCounter{}, 512)
			chunks, err := chunker.Chunk(context.Background(), tt.content)
			if err != nil {
				t.Fatalf("Chunk: %v", err)
			}
			got := make([]string, len(chunks))
			for i, c := range chunks {
				got[i] = c.Content
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("chunks = %q\nwant     %q", got, tt.want)
			}
		})
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
