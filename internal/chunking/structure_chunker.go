package chunking

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"etsi-rag/internal/config"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/tokencount"
)

var headingRe = regexp.MustCompile(`(?m)^(#{1,6})\s+(.+)$`)

// section represents a parsed markdown section with its heading hierarchy.
type section struct {
	level     int      // heading level (1-6), 0 for content before first heading
	title     string   // heading text
	body      string   // content below the heading (excluding sub-headings)
	start     int      // byte offset (in cleaned text) where the section begins
	end       int      // byte offset where the section ends (exclusive)
	imageRefs []string // refs whose page range overlaps [start, end)
}

// render reconstructs the section markdown (heading + body). Returns the
// body alone for level-0 (heading-free) sections. `body` is already trimmed
// by parseSections.
func (s section) render() string {
	if s.level == 0 {
		return s.body
	}
	out := strings.Repeat("#", s.level) + " " + s.title
	if s.body != "" {
		out += "\n\n" + s.body
	}
	return out
}

// StructureAwareChunker splits markdown text on heading boundaries, then uses
// token-based splitting only for oversized sections. Each chunk carries a
// SectionTitle breadcrumb for context.
//
// Stored chunks contain plain body (with their own h# heading markdown); the
// breadcrumb path lives on Chunk.SectionTitle. The retrieval pipeline
// prepends the breadcrumb only when handing text to the embedder, and the
// FTS5 trigger over chunks_fts concatenates section_title with content so
// BM25 still sees heading words.
type StructureAwareChunker struct {
	config             config.ChunkingConfig
	counter            tokencount.Counter
	maxEmbeddingTokens int
}

// NewStructureAwareChunker creates a structure-aware chunker.
func NewStructureAwareChunker(cfg config.ChunkingConfig, counter tokencount.Counter, maxEmbeddingTokens int) *StructureAwareChunker {
	if len(cfg.Separators) == 0 {
		cfg.Separators = DefaultChunkingConfig().Separators
	}
	return &StructureAwareChunker{
		config:             cfg,
		counter:            counter,
		maxEmbeddingTokens: maxEmbeddingTokens,
	}
}

func (c *StructureAwareChunker) effectiveMaxTokens() int {
	max := c.config.MaxChunkSize
	if c.maxEmbeddingTokens > 0 && c.maxEmbeddingTokens < max {
		max = c.maxEmbeddingTokens
	}
	return max
}

// Chunk splits content into structure-aware chunks.
func (c *StructureAwareChunker) Chunk(ctx context.Context, content string) ([]Chunk, error) {
	if content == "" {
		return nil, nil
	}
	if c.counter == nil {
		return nil, fmt.Errorf("token counter not configured")
	}

	// Pull image markers out of the input up front. The cleaned text drives all
	// section parsing; boundaries map back to which page each section's byte
	// range overlaps. Each emitted chunk inherits its source section's refs.
	cleaned, boundaries := markers.ExtractBoundaries(content)

	sections := parseSections(cleaned)
	if len(sections) == 0 {
		return nil, nil
	}
	attachImageRefs(sections, boundaries)

	// If no headings found (all content in section with level 0), fall back to token splitting
	hasHeadings := false
	for _, s := range sections {
		if s.level > 0 {
			hasHeadings = true
			break
		}
	}
	if !hasHeadings {
		return c.fallbackTokenSplit(ctx, cleaned, dedupRefs(boundaries))
	}

	maxTokens := c.effectiveMaxTokens()
	minTokens := c.config.MinChunkSize

	// Build breadcrumbs for each section
	breadcrumbs := buildBreadcrumbs(sections)

	// Process sections into chunks
	var chunks []Chunk
	idx := 0

	for i := 0; i < len(sections); i++ {
		s := sections[i]
		if s.body == "" && s.level > 0 {
			// Heading-only section with no body — skip (title will appear in child sections' breadcrumbs)
			continue
		}

		breadcrumb := breadcrumbs[i]
		refs := s.imageRefs

		// Reconstruct the section markdown (heading + body) — stored as-is.
		sectionContent := s.render()
		if sectionContent == "" {
			continue
		}

		// Token count of what the embedder will see: breadcrumb prefix + body.
		// Storage is the body only; the pipeline prepends the breadcrumb at
		// embed time. Sizing accounts for both so the embed-time payload fits.
		tokens, err := c.counter.Count(ctx, EmbeddingText(breadcrumb, sectionContent))
		if err != nil {
			return nil, fmt.Errorf("counting section tokens: %w", err)
		}

		// Try merging small sections with next sibling at same level. The
		// merged chunk is labelled by the parent breadcrumb (one level up)
		// since both siblings share the same parent — cleaner than the
		// previous "X + Y" join syntax.
		if tokens < minTokens && i+1 < len(sections) && sections[i+1].level == s.level {
			merged := sectionContent + "\n\n" + sections[i+1].render()
			mergedBreadcrumb := parentBreadcrumb(breadcrumb)
			mergedTokens, err := c.counter.Count(ctx, EmbeddingText(mergedBreadcrumb, merged))
			if err != nil {
				return nil, fmt.Errorf("counting merged tokens: %w", err)
			}
			if mergedTokens <= maxTokens {
				sectionContent = merged
				tokens = mergedTokens
				refs = unionRefs(refs, sections[i+1].imageRefs)
				breadcrumb = mergedBreadcrumb
				i++ // skip next section
			}
		}

		if tokens <= maxTokens {
			chunks = append(chunks, Chunk{
				Content:      sectionContent,
				Index:        idx,
				SectionTitle: breadcrumb,
				ImageRefs:    cloneRefs(refs),
			})
			idx++
		} else {
			// Split oversized section with token-based splitting
			subChunks, err := c.splitOversizedSection(ctx, sectionContent, breadcrumb, refs, maxTokens, &idx)
			if err != nil {
				return nil, fmt.Errorf("splitting oversized section: %w", err)
			}
			chunks = append(chunks, subChunks...)
		}
	}

	return chunks, nil
}

// parentBreadcrumb returns the breadcrumb one level up by dropping the last
// "> "-delimited segment. Returns "" for a top-level (single-segment)
// breadcrumb or an empty input.
func parentBreadcrumb(b string) string {
	if idx := strings.LastIndex(b, " > "); idx > 0 {
		return b[:idx]
	}
	return ""
}

// attachImageRefs stamps each section with the refs whose page range overlaps
// the section's [start, end) byte span in the cleaned text. A boundary at
// offset B covers the half-open range [B, nextB) (or [B, +∞) for the last).
func attachImageRefs(sections []section, boundaries []markers.Boundary) {
	if len(boundaries) == 0 {
		return
	}
	for i := range sections {
		s := &sections[i]
		seen := make(map[string]struct{})
		for j, b := range boundaries {
			rangeEnd := -1
			if j+1 < len(boundaries) {
				rangeEnd = boundaries[j+1].Start
			}
			// Section [s.start, s.end) overlaps page range [b.Start, rangeEnd)?
			if rangeEnd != -1 && rangeEnd <= s.start {
				continue
			}
			if b.Start >= s.end {
				continue
			}
			if _, dup := seen[b.Ref]; dup {
				continue
			}
			seen[b.Ref] = struct{}{}
			s.imageRefs = append(s.imageRefs, b.Ref)
		}
	}
}

// dedupRefs returns the unique ref strings from boundaries, preserving order
// of first occurrence.
func dedupRefs(boundaries []markers.Boundary) []string {
	if len(boundaries) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(boundaries))
	out := make([]string, 0, len(boundaries))
	for _, b := range boundaries {
		if _, dup := seen[b.Ref]; dup {
			continue
		}
		seen[b.Ref] = struct{}{}
		out = append(out, b.Ref)
	}
	return out
}

// unionRefs returns the dedup-union of two ref slices, preserving order
// (a's order first, then any new entries from b).
func unionRefs(a, b []string) []string {
	if len(a) == 0 {
		return cloneRefs(b)
	}
	if len(b) == 0 {
		return cloneRefs(a)
	}
	seen := make(map[string]struct{}, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, r := range a {
		if _, dup := seen[r]; dup {
			continue
		}
		seen[r] = struct{}{}
		out = append(out, r)
	}
	for _, r := range b {
		if _, dup := seen[r]; dup {
			continue
		}
		seen[r] = struct{}{}
		out = append(out, r)
	}
	return out
}

// cloneRefs returns a copy of the slice (or nil for empty input) so each chunk
// owns its own ImageRefs storage.
func cloneRefs(refs []string) []string {
	if len(refs) == 0 {
		return nil
	}
	out := make([]string, len(refs))
	copy(out, refs)
	return out
}

// parseSections splits markdown text on heading boundaries. Each returned
// section records its [start, end) byte range in `text` so callers can map
// other offset-keyed metadata (e.g. image-marker boundaries) back to sections.
func parseSections(text string) []section {
	locs := headingRe.FindAllStringIndex(text, -1)
	if len(locs) == 0 {
		// No headings found — return single untitled section
		body := strings.TrimSpace(text)
		if body == "" {
			return nil
		}
		return []section{{level: 0, body: body, start: 0, end: len(text)}}
	}

	var sections []section

	// Content before first heading
	if locs[0][0] > 0 {
		pre := strings.TrimSpace(text[:locs[0][0]])
		if pre != "" {
			sections = append(sections, section{
				level: 0, body: pre,
				start: 0, end: locs[0][0],
			})
		}
	}

	for i, loc := range locs {
		match := headingRe.FindStringSubmatch(text[loc[0]:loc[1]])
		level := len(match[1])
		title := strings.TrimSpace(match[2])

		// Body is text between this heading and the next heading (or end)
		bodyStart := loc[1]
		var bodyEnd int
		if i+1 < len(locs) {
			bodyEnd = locs[i+1][0]
		} else {
			bodyEnd = len(text)
		}
		body := strings.TrimSpace(text[bodyStart:bodyEnd])

		sections = append(sections, section{
			level: level,
			title: title,
			body:  body,
			start: loc[0],
			end:   bodyEnd,
		})
	}

	return sections
}

// buildBreadcrumbs builds heading breadcrumbs for each section based on heading hierarchy.
func buildBreadcrumbs(sections []section) []string {
	breadcrumbs := make([]string, len(sections))
	// Track the current heading at each level
	stack := make([]string, 7) // index 1-6 for h1-h6

	for i, s := range sections {
		if s.level == 0 {
			breadcrumbs[i] = ""
			continue
		}

		// Clear deeper levels when a higher-level heading appears
		for l := s.level; l <= 6; l++ {
			stack[l] = ""
		}
		stack[s.level] = s.title

		// Build breadcrumb from stack
		var parts []string
		for l := 1; l <= s.level; l++ {
			if stack[l] != "" {
				parts = append(parts, stack[l])
			}
		}
		breadcrumbs[i] = strings.Join(parts, " > ")
	}

	return breadcrumbs
}

// splitOversizedSection splits a section that exceeds maxTokens using
// separator-based splitting. The breadcrumb's token cost is reserved from
// the budget so the embed-time payload still fits the embedder's window.
func (c *StructureAwareChunker) splitOversizedSection(ctx context.Context, content, breadcrumb string, refs []string, maxTokens int, idx *int) ([]Chunk, error) {
	// Reserve budget for the embed-time breadcrumb prefix. If the breadcrumb
	// alone would leave no usable body budget, drop it for this section —
	// a chunk without a breadcrumb is still useful; a zero-body chunk is not.
	prefixTokens := 0
	if breadcrumb != "" {
		pt, err := c.counter.Count(ctx, EmbeddingText(breadcrumb, ""))
		if err != nil {
			return nil, err
		}
		prefixTokens = pt
	}
	effectiveMax := maxTokens - prefixTokens
	floor := c.config.MinChunkSize
	if floor <= 0 {
		floor = 16
	}
	if effectiveMax < floor {
		breadcrumb = ""
		effectiveMax = maxTokens
	}

	pieces, err := splitAndMerge(ctx, c.counter, content, c.config.Separators, effectiveMax, c.config.ChunkOverlap)
	if err != nil {
		return nil, err
	}

	chunks := make([]Chunk, 0, len(pieces))
	for _, p := range pieces {
		chunks = append(chunks, c.makeStructuredChunk(p, breadcrumb, refs, idx))
	}
	return chunks, nil
}

// makeStructuredChunk creates a chunk with section title and refs.
// Content is plain body — the breadcrumb is not inlined.
func (c *StructureAwareChunker) makeStructuredChunk(text, breadcrumb string, refs []string, idx *int) Chunk {
	chunk := Chunk{
		Content:      strings.TrimSpace(text),
		Index:        *idx,
		SectionTitle: breadcrumb,
		ImageRefs:    cloneRefs(refs),
	}
	*idx++
	return chunk
}

// fallbackTokenSplit splits unstructured text (no headings) using
// separator-based token splitting. Each emitted chunk inherits the entire
// input's refs (best-effort attribution since the unstructured path has no
// section boundaries to align with — user-confirmed: union all rather than
// skip refs or attempt per-chunk byte-range estimation).
func (c *StructureAwareChunker) fallbackTokenSplit(ctx context.Context, content string, refs []string) ([]Chunk, error) {
	maxTokens := c.effectiveMaxTokens()

	total, err := c.counter.Count(ctx, content)
	if err != nil {
		return nil, fmt.Errorf("counting fallback tokens: %w", err)
	}
	if total <= maxTokens {
		return stampImageRefs(
			[]Chunk{{Content: strings.TrimSpace(content), Index: 0}},
			refs,
		), nil
	}

	pieces, err := splitAndMerge(ctx, c.counter, content, c.config.Separators, maxTokens, c.config.ChunkOverlap)
	if err != nil {
		return nil, err
	}

	chunks := make([]Chunk, 0, len(pieces))
	for i, p := range pieces {
		chunks = append(chunks, Chunk{Content: p, Index: i})
	}
	return stampImageRefs(chunks, refs), nil
}

// splitBySeparators recursively splits text using the separator hierarchy.
// Stops at the first separator that produces more than one part; falls
// through to the next separator otherwise. The empty-string separator is
// treated as a no-op stop (caller is expected to follow with hard token
// splitting).
func splitBySeparators(text string, separators []string, sepIdx int) []string {
	if sepIdx >= len(separators) {
		return []string{text}
	}

	sep := separators[sepIdx]
	if sep == "" {
		return []string{text}
	}

	parts := strings.Split(text, sep)
	if len(parts) <= 1 {
		return splitBySeparators(text, separators, sepIdx+1)
	}

	var result []string
	for i, part := range parts {
		piece := part
		if i < len(parts)-1 {
			piece += sep
		}
		if strings.TrimSpace(piece) == "" {
			continue
		}
		result = append(result, piece)
	}
	return result
}

// splitAndMerge breaks text into token-bounded pieces using the separator
// hierarchy. Segments are merged greedily to fill each piece; segments that
// individually exceed maxTokens are hard-split by token range with overlap.
// Returned pieces are trimmed plain text — callers attach breadcrumb / refs.
func splitAndMerge(ctx context.Context, counter tokencount.Counter, text string, separators []string, maxTokens, overlap int) ([]string, error) {
	if maxTokens <= 0 {
		return nil, nil
	}
	segments := splitBySeparators(text, separators, 0)

	var pieces []string
	var current strings.Builder
	var currentTokens int

	flush := func() {
		if current.Len() == 0 {
			return
		}
		trimmed := strings.TrimSpace(current.String())
		if trimmed != "" {
			pieces = append(pieces, trimmed)
		}
		current.Reset()
		currentTokens = 0
	}

	for _, seg := range segments {
		segTokens, err := counter.Count(ctx, seg)
		if err != nil {
			return nil, fmt.Errorf("counting segment tokens: %w", err)
		}

		if segTokens > maxTokens {
			// Flush current then hard-split the oversized segment.
			flush()
			subs, err := hardSplitByTokens(ctx, counter, seg, maxTokens, overlap)
			if err != nil {
				return nil, err
			}
			pieces = append(pieces, subs...)
			continue
		}

		if currentTokens+segTokens > maxTokens && current.Len() > 0 {
			flush()
		}

		current.WriteString(seg)
		currentTokens += segTokens
	}
	flush()

	return pieces, nil
}

// hardSplitByTokens splits an oversized segment using tokenize/detokenize.
// Returns nil (no error) if the input is empty or maxTokens is non-positive
// — degenerate inputs would otherwise loop forever or send an empty token
// slice to the detokenizer.
func hardSplitByTokens(ctx context.Context, counter tokencount.Counter, text string, maxTokens, overlap int) ([]string, error) {
	if maxTokens <= 0 {
		return nil, nil
	}
	tokens, err := counter.Tokenize(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("tokenizing for hard split: %w", err)
	}
	if len(tokens) == 0 {
		return nil, nil
	}

	step := maxTokens
	if overlap > 0 && step > overlap {
		step -= overlap
	}

	var out []string
	for i := 0; i < len(tokens); i += step {
		end := i + maxTokens
		if end > len(tokens) {
			end = len(tokens)
		}
		chunkText, err := counter.Detokenize(ctx, tokens[i:end])
		if err != nil {
			return nil, fmt.Errorf("detokenizing chunk: %w", err)
		}
		chunkText = strings.TrimSpace(chunkText)
		if chunkText != "" {
			out = append(out, chunkText)
		}
		if end == len(tokens) {
			break
		}
	}
	return out, nil
}
