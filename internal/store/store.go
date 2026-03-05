package store

import (
	"context"
	"database/sql"

	"etsi-rag/internal/domain"
)

// SearchResult represents a search result with chunk, document, and similarity score.
type SearchResult struct {
	Chunk    domain.Chunk
	Document domain.Document
	Score    float32
}

// Store provides database operations for documents and chunks.
type Store interface {
	// Documents
	CreateDocument(ctx context.Context, doc *domain.Document) error
	GetDocument(ctx context.Context, id string) (*domain.Document, error)
	ListDocuments(ctx context.Context, lang *domain.Language) ([]domain.Document, error)
	DeleteDocument(ctx context.Context, id string) error

	// Chunks
	CreateChunks(ctx context.Context, docID string, chunks []domain.Chunk, embeddings [][]float32) error
	GetChunksByDocument(ctx context.Context, docID string) ([]domain.Chunk, error)

	// Vector search
	SearchSimilar(ctx context.Context, embedding []float32, topK int, lang *domain.Language) ([]SearchResult, error)

	// Lifecycle
	Close() error
}

// SQLiteStore implements Store using SQLite with pure Go cosine similarity search.
type SQLiteStore struct {
	db                 *sql.DB
	embeddingDimension int
}

// NewSQLiteStore creates a new SQLite store.
func NewSQLiteStore(db *sql.DB, embeddingDimension int) *SQLiteStore {
	return &SQLiteStore{
		db:                 db,
		embeddingDimension: embeddingDimension,
	}
}

// Close closes the database connection.
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}
