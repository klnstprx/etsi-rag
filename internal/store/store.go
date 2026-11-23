package store

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"etsi-rag/internal/domain"
)

// SearchResult represents a search result with chunk, document, and similarity score.
type SearchResult struct {
	Chunk    domain.Chunk
	Document domain.Document
	Score    float32
}

// SQLiteStore implements document/chunk storage using SQLite with pure Go
// cosine similarity search. The embedded embeddingCache mirrors
// chunk_embeddings in memory once Warm has been called; SearchSimilar falls
// back to a full SQL scan when the cache is unloaded.
type SQLiteStore struct {
	db           *sql.DB
	embeddingDim func() int
	cache        embeddingCache
}

// NewSQLiteStore creates a new SQLite store. The embedding cache starts
// unloaded — call Warm(ctx) to populate it (typically once at server start).
// embeddingDim is called on each validation to get the current expected
// dimension.
func NewSQLiteStore(db *sql.DB, embeddingDim func() int) *SQLiteStore {
	return &SQLiteStore{
		db:           db,
		embeddingDim: embeddingDim,
	}
}

// Close closes the database connection.
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

// checkDim rejects a vector whose length differs from the configured
// embedding dimension.
func (s *SQLiteStore) checkDim(v []float32) error {
	if dim := s.embeddingDim(); len(v) != dim {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", dim, len(v))
	}
	return nil
}

// placeholders returns n comma-separated "?" markers for an SQL IN list.
func placeholders(n int) string {
	if n == 0 {
		return ""
	}
	return strings.Repeat("?,", n-1) + "?"
}

// scanner is the Scan method shared by *sql.Row and *sql.Rows, so one row
// mapper serves both single lookups and list queries.
type scanner interface {
	Scan(dest ...any) error
}

// count returns the number of rows in table.
func (s *SQLiteStore) count(ctx context.Context, table string) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM "+table).Scan(&n); err != nil {
		return 0, fmt.Errorf("counting %s: %w", table, err)
	}
	return n, nil
}

// Warm loads every (chunk_id, doc_id, embedding) tuple from the DB
// into the in-memory cache. Idempotent — calling it again clears and reloads.
// SearchSimilar uses the cache after Warm returns; before then, it falls back
// to a full SQL scan.
//
// Returns (chunks loaded, error).
func (s *SQLiteStore) Warm(ctx context.Context) (int, error) {
	n, err := s.count(ctx, "chunk_embeddings")
	if err != nil {
		return 0, err
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT ce.chunk_id, c.doc_id, ce.embedding
		FROM chunk_embeddings ce
		JOIN chunks c ON c.id = ce.chunk_id
	`)
	if err != nil {
		return 0, fmt.Errorf("loading embeddings: %w", err)
	}
	defer rows.Close()

	expectedDim := s.embeddingDim()
	vecs := make([]cachedVec, 0, n)
	for rows.Next() {
		var v cachedVec
		var blob []byte
		if err := rows.Scan(&v.ChunkID, &v.DocID, &blob); err != nil {
			return 0, fmt.Errorf("scanning embedding row: %w", err)
		}
		v.Vec = deserializeFloat32(blob)
		if expectedDim > 0 && len(v.Vec) != expectedDim {
			return 0, fmt.Errorf("stored embedding dim mismatch for chunk %d: expected %d, got %d",
				v.ChunkID, expectedDim, len(v.Vec))
		}
		vecs = append(vecs, v)
	}
	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("iterating embeddings: %w", err)
	}

	dim := expectedDim
	if dim == 0 && len(vecs) > 0 {
		dim = len(vecs[0].Vec)
	}
	s.cache.warm(vecs, dim)
	return len(vecs), nil
}
