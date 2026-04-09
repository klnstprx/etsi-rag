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

// SearchSimilar performs brute-force cosine similarity search in Go.
func (s *SQLiteStore) SearchSimilar(ctx context.Context, embedding []float32, topK int, lang *domain.Language) ([]SearchResult, error) {
	if dim := s.embeddingDim(); len(embedding) != dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
			dim, len(embedding))
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
		if dim := s.embeddingDim(); len(vec) != dim {
			return nil, fmt.Errorf("stored embedding dimension mismatch for chunk %d: expected %d, got %d",
				chunkID, dim, len(vec))
		}
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

	// Build score lookup and placeholder list for batched fetch
	scoreMap := make(map[int64]float32, topK)
	placeholders := make([]string, topK)
	fetchArgs := make([]interface{}, topK)
	for i, sc := range topScores {
		scoreMap[sc.chunkID] = sc.score
		placeholders[i] = "?"
		fetchArgs[i] = sc.chunkID
	}

	// Fetch chunk + document metadata for all top-K IDs in one query.
	// Deliberately excludes d.content — callers only need d.id, d.title, d.url.
	fetchQuery := `
		SELECT
			c.id, c.doc_id, c.content, c.chunk_index, c.section_title,
			d.id, d.title, d.url, d.language, d.category
		FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.id IN (` + strings.Join(placeholders, ",") + `)
	`
	fetchRows, err := s.db.QueryContext(ctx, fetchQuery, fetchArgs...)
	if err != nil {
		return nil, fmt.Errorf("fetching top-K chunks: %w", err)
	}
	defer fetchRows.Close()

	results := make([]SearchResult, 0, topK)
	for fetchRows.Next() {
		var r SearchResult
		if err := fetchRows.Scan(
			&r.Chunk.ID, &r.Chunk.DocID, &r.Chunk.Content, &r.Chunk.ChunkIndex, &r.Chunk.SectionTitle,
			&r.Document.ID, &r.Document.Title, &r.Document.URL, &r.Document.Language, &r.Document.Category,
		); err != nil {
			return nil, fmt.Errorf("scanning result: %w", err)
		}
		r.Score = scoreMap[r.Chunk.ID]
		results = append(results, r)
	}
	if err := fetchRows.Err(); err != nil {
		return nil, fmt.Errorf("iterating results: %w", err)
	}

	// Re-sort by score descending — the IN query doesn't preserve order.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}
