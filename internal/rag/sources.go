package rag

import "etsi-rag/internal/domain"

// SourcesFromRetrieval builds source references from retrieval results. It
// flattens the (chunk, document) pairs back into per-chunk SourceReference
// entries that the agent and REST handlers expose as citation metadata.
func SourcesFromRetrieval(results *RetrievalResult) []domain.SourceReference {
	if results == nil {
		return nil
	}
	docMap := make(map[string]domain.Document)
	for _, d := range results.Documents {
		docMap[d.ID] = d
	}
	var sources []domain.SourceReference
	for _, sc := range results.Chunks {
		doc := docMap[sc.Chunk.DocID]
		sources = append(sources, domain.SourceReference{
			ID:         doc.ID,
			Title:      doc.Title,
			URL:        doc.URL,
			ChunkIndex: sc.Chunk.ChunkIndex,
			Score:      sc.Score,
		})
	}
	return sources
}
