// Package chunking provides document chunking strategies for RAG.
package chunking

import (
	"context"
	"fmt"
	"strings"

	"etsi-rag/internal/config"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/tokencount"
)

// stampImageRefs assigns the same ref slice to every chunk and returns the
// (possibly modified) chunks. Used by chunkers that don't track section-level
// page boundaries — they take the union of all refs in the input as a
// best-effort attribution.
func stampImageRefs(chunks []Chunk, refs []string) []Chunk {
	if len(refs) == 0 || len(chunks) == 0 {
		return chunks
	}
	for i := range chunks {
		chunks[i].ImageRefs = cloneRefs(refs)
	}
	return chunks
}

// Chunk represents a text chunk with metadata.
type Chunk struct {
	// Content is the chunk text.
	Content string

	// Index is the position within the source document (0-indexed).
	Index int

	// StartChar is the character offset where this chunk starts in the original.
	StartChar int

	// EndChar is the character offset where this chunk ends in the original.
	EndChar int

	// SectionTitle is the heading breadcrumb for structure-aware chunks (e.g. "Plan de estudios > Primer curso").
	SectionTitle string

	// ImageRefs lists image-artifact references whose page range overlaps this
	// chunk's source span (e.g. ["p1"], ["p1","p2"], ["full"]). Populated by
	// the chunker from inline [[image:ref]] markers; the markers themselves
	// are stripped from Content before emission.
	ImageRefs []string
}

// Chunker splits document content into chunks for embedding.
type Chunker interface {
	// Chunk splits content into chunks.
	// Returns nil if content is empty.
	Chunk(ctx context.Context, content string) ([]Chunk, error)
}

// DefaultChunkingConfig returns sensible defaults for chunking.
func DefaultChunkingConfig() config.ChunkingConfig {
	return config.ChunkingConfig{
		MaxChunkSize: 256,
		ChunkOverlap: 32,
		Separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}
}

// RecursiveCharacterChunker splits text recursively using a hierarchy of separators.
// Operates on character counts (no token counter needed).
type RecursiveCharacterChunker struct {
	config.ChunkingConfig
}

// NewRecursiveCharacterChunker creates a new recursive character chunker.
func NewRecursiveCharacterChunker(cfg config.ChunkingConfig) *RecursiveCharacterChunker {
	if len(cfg.Separators) == 0 {
		cfg.Separators = DefaultChunkingConfig().Separators
	}
	return &RecursiveCharacterChunker{ChunkingConfig: cfg}
}

// Chunk splits content into chunks using recursive character splitting.
func (c *RecursiveCharacterChunker) Chunk(_ context.Context, content string) ([]Chunk, error) {
	if content == "" {
		return nil, nil
	}

	cleaned, boundaries := markers.ExtractBoundaries(content)
	splits := c.splitText(cleaned, 0)
	chunks := c.mergeWithOverlap(splits, cleaned)
	return stampImageRefs(chunks, dedupRefs(boundaries)), nil
}

// splitText recursively splits text using separators at the given depth.
func (c *RecursiveCharacterChunker) splitText(text string, sepIdx int) []string {
	if len(text) <= c.MaxChunkSize {
		return []string{text}
	}

	if sepIdx >= len(c.Separators) {
		// No more separators — hard split at MaxChunkSize
		return c.hardSplit(text)
	}

	sep := c.Separators[sepIdx]

	// Empty separator means split by character
	if sep == "" {
		return c.hardSplit(text)
	}

	parts := strings.Split(text, sep)
	if len(parts) <= 1 {
		// Separator not found, try next
		return c.splitText(text, sepIdx+1)
	}

	var result []string
	var current strings.Builder

	for i, part := range parts {
		piece := part
		if i < len(parts)-1 {
			piece += sep // Re-attach separator
		}

		if current.Len()+len(piece) > c.MaxChunkSize && current.Len() > 0 {
			result = append(result, current.String())
			current.Reset()
		}

		if len(piece) > c.MaxChunkSize {
			// This piece alone exceeds chunk size — recurse with next separator
			if current.Len() > 0 {
				result = append(result, current.String())
				current.Reset()
			}
			subSplits := c.splitText(piece, sepIdx+1)
			result = append(result, subSplits...)
		} else {
			current.WriteString(piece)
		}
	}

	if current.Len() > 0 {
		result = append(result, current.String())
	}

	return result
}

// hardSplit splits text into fixed-size pieces.
func (c *RecursiveCharacterChunker) hardSplit(text string) []string {
	var result []string
	runes := []rune(text)
	for i := 0; i < len(runes); i += c.MaxChunkSize {
		end := i + c.MaxChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		result = append(result, string(runes[i:end]))
	}
	return result
}

// mergeWithOverlap creates final chunks with overlap and byte offsets.
func (c *RecursiveCharacterChunker) mergeWithOverlap(splits []string, original string) []Chunk {
	if len(splits) == 0 {
		return nil
	}

	var chunks []Chunk
	searchFrom := 0

	for i, split := range splits {
		content := strings.TrimSpace(split)
		if content == "" {
			continue
		}

		// Apply overlap: prepend tail of previous chunk
		if i > 0 && c.ChunkOverlap > 0 && len(chunks) > 0 {
			prev := chunks[len(chunks)-1].Content
			prevRunes := []rune(prev)
			if len(prevRunes) > c.ChunkOverlap {
				overlap := string(prevRunes[len(prevRunes)-c.ChunkOverlap:])
				// Only prepend if content doesn't already start with the overlap
				if !strings.HasPrefix(content, overlap) {
					content = overlap + content
				}
			}
		}

		// Find position in original text
		startChar := strings.Index(original[searchFrom:], strings.TrimSpace(split))
		if startChar >= 0 {
			startChar += searchFrom
			searchFrom = startChar + len(strings.TrimSpace(split))
		} else {
			startChar = searchFrom
		}
		endChar := startChar + len(strings.TrimSpace(split))

		chunks = append(chunks, Chunk{
			Content:   content,
			Index:     len(chunks),
			StartChar: startChar,
			EndChar:   endChar,
		})
	}

	return chunks
}

// RecursiveTokenChunker splits text recursively using token counts.
// Uses a Counter for precise tokenization and respects embedding model context windows.
type RecursiveTokenChunker struct {
	config             config.ChunkingConfig
	counter            tokencount.Counter
	maxEmbeddingTokens int
}

// NewRecursiveTokenChunker creates a token-aware chunker.
// maxEmbeddingTokens is the embedding model's context window; chunks are capped at
// min(config.MaxChunkSize, maxEmbeddingTokens).
func NewRecursiveTokenChunker(cfg config.ChunkingConfig, counter tokencount.Counter, maxEmbeddingTokens int) *RecursiveTokenChunker {
	if len(cfg.Separators) == 0 {
		cfg.Separators = DefaultChunkingConfig().Separators
	}
	return &RecursiveTokenChunker{
		config:             cfg,
		counter:            counter,
		maxEmbeddingTokens: maxEmbeddingTokens,
	}
}

func (c *RecursiveTokenChunker) effectiveMaxTokens() int {
	max := c.config.MaxChunkSize
	if c.maxEmbeddingTokens > 0 && c.maxEmbeddingTokens < max {
		max = c.maxEmbeddingTokens
	}
	return max
}

// Chunk splits content into token-aware chunks.
func (c *RecursiveTokenChunker) Chunk(ctx context.Context, content string) ([]Chunk, error) {
	if content == "" {
		return nil, nil
	}
	if c.counter == nil {
		return nil, fmt.Errorf("token counter not configured")
	}

	cleaned, boundaries := markers.ExtractBoundaries(content)
	refs := dedupRefs(boundaries)

	maxTokens := c.effectiveMaxTokens()

	totalTokens, err := c.counter.Count(ctx, cleaned)
	if err != nil {
		return nil, fmt.Errorf("counting tokens: %w", err)
	}

	// If it fits in one chunk, return immediately
	if totalTokens <= maxTokens {
		return stampImageRefs(
			[]Chunk{{Content: cleaned, Index: 0, StartChar: 0, EndChar: len(cleaned)}},
			refs,
		), nil
	}

	// Split text by separators (character-level), then merge respecting token limits
	segments := c.splitBySeparators(cleaned, 0)
	chunks, err := c.mergeSegments(ctx, segments, cleaned, maxTokens)
	if err != nil {
		return nil, err
	}
	return stampImageRefs(chunks, refs), nil
}

// splitBySeparators recursively splits text using the separator hierarchy.
func (c *RecursiveTokenChunker) splitBySeparators(text string, sepIdx int) []string {
	if sepIdx >= len(c.config.Separators) {
		return []string{text}
	}

	sep := c.config.Separators[sepIdx]
	if sep == "" {
		return []string{text}
	}

	parts := strings.Split(text, sep)
	if len(parts) <= 1 {
		return c.splitBySeparators(text, sepIdx+1)
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

// mergeSegments merges text segments into chunks that respect token limits.
func (c *RecursiveTokenChunker) mergeSegments(ctx context.Context, segments []string, original string, maxTokens int) ([]Chunk, error) {
	var chunks []Chunk
	var current strings.Builder
	var currentTokens int
	searchFrom := 0

	for _, seg := range segments {
		segTokens, err := c.counter.Count(ctx, seg)
		if err != nil {
			return nil, fmt.Errorf("counting segment tokens: %w", err)
		}

		// If this single segment exceeds max, split it further
		if segTokens > maxTokens {
			// Flush current buffer
			if current.Len() > 0 {
				chunks = append(chunks, c.makeChunk(ctx, current.String(), original, &searchFrom, len(chunks), chunks))
				current.Reset()
				currentTokens = 0
			}
			// Hard split the oversized segment by tokens
			subChunks, err := c.hardSplitByTokens(ctx, seg, original, &searchFrom, len(chunks), maxTokens, chunks)
			if err != nil {
				return nil, err
			}
			chunks = append(chunks, subChunks...)
			continue
		}

		// Would adding this segment exceed the limit?
		if currentTokens+segTokens > maxTokens && current.Len() > 0 {
			chunks = append(chunks, c.makeChunk(ctx, current.String(), original, &searchFrom, len(chunks), chunks))
			current.Reset()
			currentTokens = 0
		}

		current.WriteString(seg)
		currentTokens += segTokens
	}

	if current.Len() > 0 {
		chunks = append(chunks, c.makeChunk(ctx, current.String(), original, &searchFrom, len(chunks), chunks))
	}

	return chunks, nil
}

// hardSplitByTokens splits an oversized segment using tokenize/detokenize.
// Returns an empty slice (no error) if the input is empty or maxTokens is
// non-positive — degenerate inputs would otherwise cause an infinite loop or
// send an empty slice to the detokenizer.
func (c *RecursiveTokenChunker) hardSplitByTokens(ctx context.Context, text, original string, searchFrom *int, startIdx, maxTokens int, prevChunks []Chunk) ([]Chunk, error) {
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
	step := maxTokens - c.config.ChunkOverlap
	if step <= 0 {
		step = maxTokens
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

		allPrev := append(prevChunks, chunks...)
		chunk := c.makeChunk(ctx, chunkText, original, searchFrom, startIdx+len(chunks), allPrev)
		chunks = append(chunks, chunk)

		if end == len(tokens) {
			break
		}
	}

	return chunks, nil
}

// makeChunk constructs a Chunk with overlap and position tracking.
func (c *RecursiveTokenChunker) makeChunk(ctx context.Context, text, original string, searchFrom *int, index int, prevChunks []Chunk) Chunk {
	content := strings.TrimSpace(text)

	// Apply token overlap from previous chunk
	if index > 0 && c.config.ChunkOverlap > 0 && len(prevChunks) > 0 {
		prev := prevChunks[len(prevChunks)-1].Content
		prevTokens, err := c.counter.Tokenize(ctx, prev)
		if err == nil && len(prevTokens) > c.config.ChunkOverlap {
			overlapTokens := prevTokens[len(prevTokens)-c.config.ChunkOverlap:]
			overlap, err := c.counter.Detokenize(ctx, overlapTokens)
			if err == nil && overlap != "" && !strings.HasPrefix(content, overlap) {
				content = overlap + content
			}
		}
	}

	// Find position in original text
	trimmed := strings.TrimSpace(text)
	startChar := strings.Index(original[*searchFrom:], trimmed)
	if startChar >= 0 {
		startChar += *searchFrom
		*searchFrom = startChar + len(trimmed)
	} else {
		startChar = *searchFrom
	}
	endChar := startChar + len(trimmed)

	return Chunk{
		Content:   content,
		Index:     index,
		StartChar: startChar,
		EndChar:   endChar,
	}
}
