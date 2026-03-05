package store

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"

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

// CreateChunks inserts chunks with their embeddings.
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
		INSERT INTO chunks (doc_id, content, chunk_index)
		VALUES (?, ?, ?)
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

	for i, chunk := range chunks {
		if len(embeddings[i]) != s.embeddingDimension {
			return fmt.Errorf("embedding dimension mismatch at index %d: expected %d, got %d",
				i, s.embeddingDimension, len(embeddings[i]))
		}

		result, err := chunkStmt.ExecContext(ctx, docID, chunk.Content, chunk.ChunkIndex)
		if err != nil {
			return fmt.Errorf("inserting chunk %d: %w", i, err)
		}

		chunkID, err := result.LastInsertId()
		if err != nil {
			return fmt.Errorf("getting chunk ID: %w", err)
		}

		blob := serializeFloat32(embeddings[i])
		if _, err := embStmt.ExecContext(ctx, chunkID, blob); err != nil {
			return fmt.Errorf("inserting embedding %d: %w", i, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	return nil
}

// GetChunksByDocument retrieves all chunks for a document.
func (s *SQLiteStore) GetChunksByDocument(ctx context.Context, docID string) ([]domain.Chunk, error) {
	query := `
		SELECT id, doc_id, content, chunk_index
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
		if err := rows.Scan(&chunk.ID, &chunk.DocID, &chunk.Content, &chunk.ChunkIndex); err != nil {
			return nil, fmt.Errorf("scanning chunk: %w", err)
		}
		chunks = append(chunks, chunk)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating chunks: %w", err)
	}

	return chunks, nil
}

// SearchSimilar performs brute-force cosine similarity search in Go.
func (s *SQLiteStore) SearchSimilar(ctx context.Context, embedding []float32, topK int, lang *domain.Language) ([]SearchResult, error) {
	if len(embedding) != s.embeddingDimension {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
			s.embeddingDimension, len(embedding))
	}

	// Load all embeddings (with optional language filter via join)
	var query string
	var args []interface{}

	if lang != nil {
		query = `
			SELECT ce.chunk_id, ce.embedding
			FROM chunk_embeddings ce
			JOIN chunks c ON c.id = ce.chunk_id
			JOIN documents d ON d.id = c.doc_id
			WHERE d.language = ?
		`
		args = []interface{}{*lang}
	} else {
		query = `SELECT chunk_id, embedding FROM chunk_embeddings`
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("loading embeddings: %w", err)
	}
	defer rows.Close()

	// Compute similarities
	type scored struct {
		chunkID int64
		score   float32
	}
	var scores []scored

	for rows.Next() {
		var chunkID int64
		var blob []byte
		if err := rows.Scan(&chunkID, &blob); err != nil {
			return nil, fmt.Errorf("scanning embedding: %w", err)
		}
		vec := deserializeFloat32(blob)
		sim := cosineSimilarity(embedding, vec)
		scores = append(scores, scored{chunkID: chunkID, score: sim})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating embeddings: %w", err)
	}

	if len(scores) == 0 {
		return nil, nil
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Take top-K
	if topK > len(scores) {
		topK = len(scores)
	}
	topScores := scores[:topK]

	// Fetch chunk + document data for top-K IDs
	results := make([]SearchResult, 0, topK)
	fetchQuery := `
		SELECT
			c.id, c.doc_id, c.content, c.chunk_index,
			d.id, d.title, d.content, d.url, d.language, d.category
		FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.id = ?
	`
	for _, sc := range topScores {
		var r SearchResult
		scanErr := s.db.QueryRowContext(ctx, fetchQuery, sc.chunkID).Scan(
			&r.Chunk.ID, &r.Chunk.DocID, &r.Chunk.Content, &r.Chunk.ChunkIndex,
			&r.Document.ID, &r.Document.Title, &r.Document.Content, &r.Document.URL, &r.Document.Language, &r.Document.Category,
		)
		if scanErr != nil {
			return nil, fmt.Errorf("fetching chunk %d: %w", sc.chunkID, scanErr)
		}
		r.Score = sc.score
		results = append(results, r)
	}

	return results, nil
}
