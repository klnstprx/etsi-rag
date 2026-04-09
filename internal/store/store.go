package store

import (
	"database/sql"

	"etsi-rag/internal/domain"
)

// SearchResult represents a search result with chunk, document, and similarity score.
type SearchResult struct {
	Chunk    domain.Chunk
	Document domain.Document
	Score    float32
}

// SQLiteStore implements document/chunk storage using SQLite with pure Go cosine similarity search.
type SQLiteStore struct {
	db           *sql.DB
	embeddingDim func() int
}

// NewSQLiteStore creates a new SQLite store.
// embeddingDim is called on each validation to get the current expected dimension.
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
