package store

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"

	"etsi-rag/internal/domain"
)

// serializeFloat32 converts a float32 slice to a little-endian byte slice.
func serializeFloat32(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// deserializeFloat32 converts a little-endian byte slice to a float32 slice.
func deserializeFloat32(buf []byte) []float32 {
	n := len(buf) / 4
	v := make([]float32, n)
	for i := range n {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return v
}

// cosineSimilarity computes cosine similarity between two vectors.
// Returns 0 if either vector has zero magnitude.
func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}

// CreateChunks inserts chunks with their embeddings, then mirrors the new
// entries into the in-memory cache (no-op when the cache is unloaded).
func (s *SQLiteStore) CreateChunks(ctx context.Context, docID string, chunks []domain.Chunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunks and embeddings count mismatch: %d vs %d", len(chunks), len(embeddings))
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	chunkStmt, err := tx.PrepareContext(ctx, `
		INSERT INTO chunks (doc_id, content, chunk_index, section_title)
		VALUES (?, ?, ?, ?)
	`)
	if err != nil {
		return fmt.Errorf("preparing chunk statement: %w", err)
	}
	defer chunkStmt.Close()

	embStmt, err := tx.PrepareContext(ctx, `
		INSERT INTO chunk_embeddings (chunk_id, embedding)
		VALUES (?, ?)
	`)
	if err != nil {
		return fmt.Errorf("preparing embedding statement: %w", err)
	}
	defer embStmt.Close()

	chunkIDs := make([]int64, len(chunks))
	for i, chunk := range chunks {
		if dim := s.embeddingDim(); len(embeddings[i]) != dim {
			return fmt.Errorf("embedding dimension mismatch at index %d: expected %d, got %d",
				i, dim, len(embeddings[i]))
		}

		result, err := chunkStmt.ExecContext(ctx, docID, chunk.Content, chunk.ChunkIndex, chunk.SectionTitle)
		if err != nil {
			return fmt.Errorf("inserting chunk %d: %w", i, err)
		}

		chunkID, err := result.LastInsertId()
		if err != nil {
			return fmt.Errorf("getting chunk ID: %w", err)
		}
		chunkIDs[i] = chunkID

		blob := serializeFloat32(embeddings[i])
		if _, err := embStmt.ExecContext(ctx, chunkID, blob); err != nil {
			return fmt.Errorf("inserting embedding %d: %w", i, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	entries := make([]cachedVec, len(chunks))
	for i := range chunks {
		entries[i] = cachedVec{
			ChunkID: chunkIDs[i],
			DocID:   docID,
			Vec:     embeddings[i],
		}
	}
	if err := s.cache.appendChunks(entries); err != nil {
		return fmt.Errorf("appending to cache: %w", err)
	}

	return nil
}

// GetChunksByDocument retrieves all chunks for a document.
func (s *SQLiteStore) GetChunksByDocument(ctx context.Context, docID string) ([]domain.Chunk, error) {
	query := `
		SELECT id, doc_id, content, chunk_index, section_title
		FROM chunks
		WHERE doc_id = ?
		ORDER BY chunk_index
	`
	rows, err := s.db.QueryContext(ctx, query, docID)
	if err != nil {
		return nil, fmt.Errorf("querying chunks: %w", err)
	}
	defer rows.Close()

	var chunks []domain.Chunk
	for rows.Next() {
		var chunk domain.Chunk
		if err := rows.Scan(&chunk.ID, &chunk.DocID, &chunk.Content, &chunk.ChunkIndex, &chunk.SectionTitle); err != nil {
			return nil, fmt.Errorf("scanning chunk: %w", err)
		}
		chunks = append(chunks, chunk)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating chunks: %w", err)
	}

	return chunks, nil
}

// SearchSimilar performs brute-force cosine similarity search. When the
// in-memory cache is loaded (Warm has run), scoring happens against
// pre-deserialized vectors and only the metadata for the top-K hits is
// fetched from SQLite. When the cache is unloaded, falls back to the full
// SQL scan path — keeping :memory: tests that never call Warm working.
func (s *SQLiteStore) SearchSimilar(ctx context.Context, embedding []float32, topK int) ([]SearchResult, error) {
	if dim := s.embeddingDim(); len(embedding) != dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
			dim, len(embedding))
	}

	scored, useCache, err := s.cache.searchTopK(embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("scoring against cache: %w", err)
	}
	if !useCache {
		return s.searchSimilarSQL(ctx, embedding, topK)
	}
	if len(scored) == 0 {
		return nil, nil
	}

	return s.fetchSearchResults(ctx, scored)
}

// fetchSearchResults loads chunk + document metadata for the given top-K
// scored chunk IDs in a single batched query, preserving score-descending
// order.
func (s *SQLiteStore) fetchSearchResults(ctx context.Context, scored []cacheScored) ([]SearchResult, error) {
	scoreMap := make(map[int64]float32, len(scored))
	placeholders := make([]string, len(scored))
	fetchArgs := make([]any, len(scored))
	for i, sc := range scored {
		scoreMap[sc.ChunkID] = sc.Score
		placeholders[i] = "?"
		fetchArgs[i] = sc.ChunkID
	}

	fetchQuery := `
		SELECT
			c.id, c.doc_id, c.content, c.chunk_index, c.section_title,
			d.id, d.title, d.url, d.language, d.category
		FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.id IN (` + strings.Join(placeholders, ",") + `)
	`
	rows, err := s.db.QueryContext(ctx, fetchQuery, fetchArgs...)
	if err != nil {
		return nil, fmt.Errorf("fetching top-K chunks: %w", err)
	}
	defer rows.Close()

	results := make([]SearchResult, 0, len(scored))
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(
			&r.Chunk.ID, &r.Chunk.DocID, &r.Chunk.Content, &r.Chunk.ChunkIndex, &r.Chunk.SectionTitle,
			&r.Document.ID, &r.Document.Title, &r.Document.URL, &r.Document.Language, &r.Document.Category,
		); err != nil {
			return nil, fmt.Errorf("scanning result: %w", err)
		}
		r.Score = scoreMap[r.Chunk.ID]
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating results: %w", err)
	}

	// Re-sort by score descending — the IN query doesn't preserve order.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// searchSimilarSQL is the legacy full-scan path. Used when the cache hasn't
// been warmed (typical for :memory: tests). Loads every embedding row,
// deserializes, scores, sorts, then fetches metadata for the top-K.
func (s *SQLiteStore) searchSimilarSQL(ctx context.Context, embedding []float32, topK int) ([]SearchResult, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT chunk_id, embedding FROM chunk_embeddings`)
	if err != nil {
		return nil, fmt.Errorf("loading embeddings: %w", err)
	}
	defer rows.Close()

	var scored []cacheScored
	for rows.Next() {
		var chunkID int64
		var blob []byte
		if err := rows.Scan(&chunkID, &blob); err != nil {
			return nil, fmt.Errorf("scanning embedding: %w", err)
		}
		vec := deserializeFloat32(blob)
		if dim := s.embeddingDim(); len(vec) != dim {
			return nil, fmt.Errorf("stored embedding dimension mismatch for chunk %d: expected %d, got %d",
				chunkID, dim, len(vec))
		}
		scored = append(scored, cacheScored{ChunkID: chunkID, Score: cosineSimilarity(embedding, vec)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating embeddings: %w", err)
	}

	if len(scored) == 0 {
		return nil, nil
	}

	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topK > len(scored) {
		topK = len(scored)
	}
	return s.fetchSearchResults(ctx, scored[:topK])
}
