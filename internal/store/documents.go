package store

import (
	"context"
	"database/sql"
	"fmt"

	"etsi-rag/internal/domain"
)

// CreateDocument inserts a new document.
func (s *SQLiteStore) CreateDocument(ctx context.Context, doc *domain.Document) error {
	query := `
		INSERT INTO documents (id, title, content, url, language, category)
		VALUES (?, ?, ?, ?, ?, ?)
	`
	_, err := s.db.ExecContext(ctx, query,
		doc.ID, doc.Title, doc.Content, doc.URL, doc.Language, doc.Category,
	)
	if err != nil {
		return fmt.Errorf("inserting document: %w", err)
	}
	return nil
}

// GetDocument retrieves a document by ID.
func (s *SQLiteStore) GetDocument(ctx context.Context, id string) (*domain.Document, error) {
	query := `
		SELECT id, title, content, url, language, category
		FROM documents
		WHERE id = ?
	`
	var doc domain.Document
	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&doc.ID, &doc.Title, &doc.Content, &doc.URL, &doc.Language, &doc.Category,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("querying document: %w", err)
	}
	return &doc, nil
}

// ListDocuments retrieves all documents, optionally filtered by language.
func (s *SQLiteStore) ListDocuments(ctx context.Context, lang *domain.Language) ([]domain.Document, error) {
	var rows *sql.Rows
	var err error

	if lang != nil {
		query := `
			SELECT id, title, content, url, language, category
			FROM documents
			WHERE language = ?
			ORDER BY created_at DESC
		`
		rows, err = s.db.QueryContext(ctx, query, *lang)
	} else {
		query := `
			SELECT id, title, content, url, language, category
			FROM documents
			ORDER BY created_at DESC
		`
		rows, err = s.db.QueryContext(ctx, query)
	}

	if err != nil {
		return nil, fmt.Errorf("querying documents: %w", err)
	}
	defer rows.Close()

	var docs []domain.Document
	for rows.Next() {
		var doc domain.Document
		if err := rows.Scan(
			&doc.ID, &doc.Title, &doc.Content, &doc.URL, &doc.Language, &doc.Category,
		); err != nil {
			return nil, fmt.Errorf("scanning document: %w", err)
		}
		docs = append(docs, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating documents: %w", err)
	}

	return docs, nil
}

// DeleteDocument deletes a document and its chunks + embeddings (via CASCADE).
func (s *SQLiteStore) DeleteDocument(ctx context.Context, id string) error {
	// Delete document (chunks and embeddings are deleted via CASCADE)
	deleteDocQuery := `DELETE FROM documents WHERE id = ?`
	result, err := s.db.ExecContext(ctx, deleteDocQuery, id)
	if err != nil {
		return fmt.Errorf("deleting document: %w", err)
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("checking rows affected: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("document not found: %s", id)
	}

	return nil
}
