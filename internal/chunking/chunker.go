// Package chunking provides document chunking strategies for RAG.
package chunking

import (
	"context"

	"etsi-rag/internal/config"
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
	// Content is the chunk text (plain body; no breadcrumb prefix).
	Content string

	// Index is the position within the source document (0-indexed).
	Index int

	// SectionTitle is the heading breadcrumb for structure-aware chunks
	// (e.g. "Plan de estudios > Primer curso"). Empty for chunks emitted
	// from the heading-free fallback path.
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

// EmbeddingText returns the text the embedder should see for a chunk with
// the given breadcrumb and body. Stored chunk content is plain body; this
// helper applies the breadcrumb prefix at embed time. The FTS5 trigger over
// chunks_fts mirrors this concatenation so BM25 also sees heading words.
// Single source of truth — shared by the chunker (for sizing) and the
// retrieval pipeline (for the actual embed-time payload).
func EmbeddingText(breadcrumb, body string) string {
	if breadcrumb == "" {
		return body
	}
	return breadcrumb + ":\n" + body
}

// DefaultChunkingConfig returns sensible defaults for chunking.
func DefaultChunkingConfig() config.ChunkingConfig {
	return config.ChunkingConfig{
		MaxChunkSize: 256,
		ChunkOverlap: 32,
		Separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}
}
