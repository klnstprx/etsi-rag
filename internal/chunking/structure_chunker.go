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

// StructureAwareChunker splits markdown text on heading boundaries, then uses
// token-based splitting only for oversized sections. Each chunk carries a
// SectionTitle breadcrumb for context.
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
		body := strings.TrimSpace(s.body)
		if body == "" && s.level > 0 {
			// Heading-only section with no body — skip (title will appear in child sections' breadcrumbs)
			continue
		}

		breadcrumb := breadcrumbs[i]
		refs := s.imageRefs

		// Build full section content (heading + body)
		var sectionContent string
		if s.level > 0 {
			sectionContent = strings.Repeat("#", s.level) + " " + s.title + "\n\n" + body
		} else {
			sectionContent = body
		}
		sectionContent = strings.TrimSpace(sectionContent)
		if sectionContent == "" {
			continue
		}

		tokens, err := c.counter.Count(ctx, sectionContent)
		if err != nil {
			return nil, fmt.Errorf("counting section tokens: %w", err)
		}

		// Try merging small sections with next sibling at same level
		if tokens < minTokens && i+1 < len(sections) && sections[i+1].level == s.level {
			nextBody := strings.TrimSpace(sections[i+1].body)
			var nextContent string
			if sections[i+1].level > 0 {
				nextContent = strings.Repeat("#", sections[i+1].level) + " " + sections[i+1].title + "\n\n" + nextBody
			} else {
				nextContent = nextBody
			}
			merged := sectionContent + "\n\n" + strings.TrimSpace(nextContent)
			mergedTokens, err := c.counter.Count(ctx, merged)
			if err != nil {
				return nil, fmt.Errorf("counting merged tokens: %w", err)
			}
			if mergedTokens <= maxTokens {
				// Merge: replace current content and skip next section
				sectionContent = merged
				tokens = mergedTokens
				refs = unionRefs(refs, sections[i+1].imageRefs)
				// Combine breadcrumbs
				if breadcrumbs[i+1] != "" && breadcrumbs[i+1] != breadcrumb {
					breadcrumb = breadcrumb + " + " + sections[i+1].title
				}
				i++ // skip next section
			}
		}

		if tokens <= maxTokens {
			// Prepend section title for embedding context
			embeddingContent := sectionContent
			if breadcrumb != "" {
				embeddingContent = breadcrumb + ":\n" + sectionContent
			}
			chunks = append(chunks, Chunk{
				Content:      embeddingContent,
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

// splitOversizedSection splits a section that exceeds maxTokens using separator-based splitting.
func (c *StructureAwareChunker) splitOversizedSection(ctx context.Context, content, breadcrumb string, refs []string, maxTokens int, idx *int) ([]Chunk, error) {
	// Use separator-based splitting (paragraph, newline, sentence, word)
	segments := splitBySeparators(content, c.config.Separators, 0)

	// Account for breadcrumb prefix in token budget. If the breadcrumb alone
	// would leave no usable room, drop it for this section — a chunk without
	// the breadcrumb is still useful; a zero-content chunk is not.
	prefixTokens := 0
	if breadcrumb != "" {
		pt, err := c.counter.Count(ctx, breadcrumb+":\n")
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
		// Breadcrumb consumes too much of the budget — drop it for this section.
		// Chunks still carry SectionTitle via makeStructuredChunk's idx bookkeeping
		// when breadcrumb is "" (no inline prefix, empty SectionTitle).
		breadcrumb = ""
		prefixTokens = 0
		effectiveMax = maxTokens
	}

	var chunks []Chunk
	var current strings.Builder
	var currentTokens int

	for _, seg := range segments {
		segTokens, err := c.counter.Count(ctx, seg)
		if err != nil {
			return nil, fmt.Errorf("counting segment tokens: %w", err)
		}

		if segTokens > effectiveMax {
			// Flush current
			if current.Len() > 0 {
				chunks = append(chunks, c.makeStructuredChunk(current.String(), breadcrumb, refs, idx))
				current.Reset()
				currentTokens = 0
			}
			// Hard split oversized segment
			subChunks, err := c.hardSplitByTokens(ctx, seg, breadcrumb, refs, effectiveMax, idx)
			if err != nil {
				return nil, err
			}
			chunks = append(chunks, subChunks...)
			continue
		}

		if currentTokens+segTokens > effectiveMax && current.Len() > 0 {
			chunks = append(chunks, c.makeStructuredChunk(current.String(), breadcrumb, refs, idx))
			current.Reset()
			currentTokens = 0
		}

		current.WriteString(seg)
		currentTokens += segTokens
	}

	if current.Len() > 0 {
		chunks = append(chunks, c.makeStructuredChunk(current.String(), breadcrumb, refs, idx))
	}

	return chunks, nil
}

// splitBySeparators recursively splits text using the separator hierarchy.
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

// makeStructuredChunk creates a chunk with section title and breadcrumb prefix.
func (c *StructureAwareChunker) makeStructuredChunk(text, breadcrumb string, refs []string, idx *int) Chunk {
	content := strings.TrimSpace(text)
	if breadcrumb != "" {
		content = breadcrumb + ":\n" + content
	}
	chunk := Chunk{
		Content:      content,
		Index:        *idx,
		SectionTitle: breadcrumb,
		ImageRefs:    cloneRefs(refs),
	}
	*idx++
	return chunk
}

// hardSplitByTokens splits an oversized segment using tokenize/detokenize.
// Returns an empty slice (no error) if the input is empty or maxTokens is
// non-positive — degenerate inputs would otherwise cause an infinite loop or
// send an empty slice to the detokenizer.
func (c *StructureAwareChunker) hardSplitByTokens(ctx context.Context, text, breadcrumb string, refs []string, maxTokens int, idx *int) ([]Chunk, error) {
	if maxTokens <= 0 {
		return nil, nil
	}
	tokens, err := c.counter.Tokenize(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("tokenizing for hard split: %w", err)
	}
	if len(tokens) == 0 {
		return nil, nil
	}

	var chunks []Chunk
	step := maxTokens
	if c.config.ChunkOverlap > 0 && step > c.config.ChunkOverlap {
		step -= c.config.ChunkOverlap
	}

	for i := 0; i < len(tokens); i += step {
		end := i + maxTokens
		if end > len(tokens) {
			end = len(tokens)
		}

		chunkText, err := c.counter.Detokenize(ctx, tokens[i:end])
		if err != nil {
			return nil, fmt.Errorf("detokenizing chunk: %w", err)
		}
		chunkText = strings.TrimSpace(chunkText)
		if chunkText == "" {
			continue
		}

		chunks = append(chunks, c.makeStructuredChunk(chunkText, breadcrumb, refs, idx))

		if end == len(tokens) {
			break
		}
	}

	return chunks, nil
}

// fallbackTokenSplit splits unstructured text (no headings) using separator-based token splitting.
// Each emitted chunk inherits the entire input's refs (best-effort attribution
// since the unstructured path has no section boundaries to align with).
func (c *StructureAwareChunker) fallbackTokenSplit(ctx context.Context, content string, refs []string) ([]Chunk, error) {
	tokenChunker := NewRecursiveTokenChunker(c.config, c.counter, c.maxEmbeddingTokens)
	chunks, err := tokenChunker.Chunk(ctx, content)
	if err != nil {
		return nil, err
	}
	if len(refs) == 0 {
		return chunks, nil
	}
	for i := range chunks {
		chunks[i].ImageRefs = cloneRefs(refs)
	}
	return chunks, nil
}
