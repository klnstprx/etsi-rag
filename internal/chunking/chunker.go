// Package chunking provides document chunking strategies for RAG.
package chunking

import (
	"strings"

	"etsi-rag/internal/config"
)

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
}

// Chunker splits document content into chunks for embedding.
type Chunker interface {
	// Chunk splits content into chunks.
	// Returns nil if content is empty.
	Chunk(content string) []Chunk
}

// DefaultChunkingConfig returns sensible defaults for chunking.
func DefaultChunkingConfig() config.ChunkingConfig {
	return config.ChunkingConfig{
		ChunkSize:    500,
		ChunkOverlap: 50,
		Separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}
}

// RecursiveCharacterChunker splits text recursively using a hierarchy of separators.
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
func (c *RecursiveCharacterChunker) Chunk(content string) []Chunk {
	if content == "" {
		return nil
	}

	splits := c.splitText(content, 0)
	return c.mergeWithOverlap(splits, content)
}

// splitText recursively splits text using separators at the given depth.
func (c *RecursiveCharacterChunker) splitText(text string, sepIdx int) []string {
	if len(text) <= c.ChunkSize {
		return []string{text}
	}

	if sepIdx >= len(c.Separators) {
		// No more separators — hard split at ChunkSize
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

		if current.Len()+len(piece) > c.ChunkSize && current.Len() > 0 {
			result = append(result, current.String())
			current.Reset()
		}

		if len(piece) > c.ChunkSize {
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
	for i := 0; i < len(runes); i += c.ChunkSize {
		end := i + c.ChunkSize
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

// TokenChunker splits text by token count (requires tokenizer).
type TokenChunker struct {
	ChunkSize    int
	ChunkOverlap int
	Tokenizer    func(string) []string
}

// NewTokenChunker creates a new token-based chunker.
func NewTokenChunker(chunkSize, chunkOverlap int, tokenizer func(string) []string) *TokenChunker {
	return &TokenChunker{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
		Tokenizer:    tokenizer,
	}
}

// Chunk splits content into chunks by token count.
func (c *TokenChunker) Chunk(content string) []Chunk {
	if content == "" {
		return nil
	}

	tokenize := c.Tokenizer
	if tokenize == nil {
		tokenize = func(s string) []string { return strings.Fields(s) }
	}

	tokens := tokenize(content)
	if len(tokens) == 0 {
		return nil
	}

	var chunks []Chunk
	step := c.ChunkSize - c.ChunkOverlap
	if step <= 0 {
		step = 1
	}

	for i := 0; i < len(tokens); i += step {
		end := i + c.ChunkSize
		if end > len(tokens) {
			end = len(tokens)
		}
		text := strings.Join(tokens[i:end], " ")
		chunks = append(chunks, Chunk{
			Content: text,
			Index:   len(chunks),
		})
		if end == len(tokens) {
			break
		}
	}

	return chunks
}

// SentenceChunker groups sentences until reaching the target size.
type SentenceChunker struct {
	MaxChunkSize int
	MinChunkSize int
}

// NewSentenceChunker creates a new sentence-based chunker.
func NewSentenceChunker(maxSize, minSize int) *SentenceChunker {
	return &SentenceChunker{
		MaxChunkSize: maxSize,
		MinChunkSize: minSize,
	}
}

// Chunk splits content into sentence-based chunks.
func (c *SentenceChunker) Chunk(content string) []Chunk {
	if content == "" {
		return nil
	}

	sentences := splitSentences(content)
	if len(sentences) == 0 {
		return nil
	}

	var chunks []Chunk
	var current strings.Builder

	for _, sent := range sentences {
		if current.Len()+len(sent) > c.MaxChunkSize && current.Len() >= c.MinChunkSize {
			chunks = append(chunks, Chunk{
				Content: strings.TrimSpace(current.String()),
				Index:   len(chunks),
			})
			current.Reset()
		}
		current.WriteString(sent)
	}

	if current.Len() > 0 {
		text := strings.TrimSpace(current.String())
		if text != "" {
			chunks = append(chunks, Chunk{
				Content: text,
				Index:   len(chunks),
			})
		}
	}

	return chunks
}

// splitSentences splits text at sentence boundaries.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// Check if followed by space or end of text
			if i+1 >= len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n' {
				sentences = append(sentences, current.String())
				current.Reset()
			}
		}
	}

	if current.Len() > 0 {
		sentences = append(sentences, current.String())
	}

	return sentences
}
