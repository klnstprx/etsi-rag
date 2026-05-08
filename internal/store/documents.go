package store

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"etsi-rag/internal/domain"
)

// CreateDocument inserts a new document.
func (s *SQLiteStore) CreateDocument(ctx context.Context, doc *domain.Document) error {
	query := `
		INSERT INTO documents (id, title, content, url, language)
		VALUES (?, ?, ?, ?, ?)
	`
	_, err := s.db.ExecContext(ctx, query,
		doc.ID, doc.Title, doc.Content, doc.URL, doc.Language,
	)
	if err != nil {
		return fmt.Errorf("inserting document: %w", err)
	}
	return nil
}

// DocumentExists reports whether a document with the given ID exists.
// Cheaper than GetDocument because it does not fetch the content column.
func (s *SQLiteStore) DocumentExists(ctx context.Context, id string) (bool, error) {
	var n int
	err := s.db.QueryRowContext(ctx, `SELECT 1 FROM documents WHERE id = ? LIMIT 1`, id).Scan(&n)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("checking document existence: %w", err)
	}
	return true, nil
}

// GetDocumentURL returns the stored URL for a document by ID. Used by the
// crawler's dedup path to pick the more canonical URL when several URLs
// resolve to the same content_hash. Returns empty string if the doc has no
// URL recorded.
func (s *SQLiteStore) GetDocumentURL(ctx context.Context, id string) (string, error) {
	var url sql.NullString
	err := s.db.QueryRowContext(ctx, `SELECT url FROM documents WHERE id = ? LIMIT 1`, id).Scan(&url)
	if err == sql.ErrNoRows {
		return "", fmt.Errorf("document %s: %w", id, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("querying document url: %w", err)
	}
	if !url.Valid {
		return "", nil
	}
	return url.String, nil
}

// UpdateDocumentURL changes the URL recorded for an existing document.
// Used by the crawler when a content-hash dedup discovers a more canonical
// URL than the one originally stored (e.g. the seed URL beating a
// `/cms/menu/...` sub-page that won the timing race).
func (s *SQLiteStore) UpdateDocumentURL(ctx context.Context, id, url string) error {
	res, err := s.db.ExecContext(ctx, `UPDATE documents SET url = ? WHERE id = ?`, url, id)
	if err != nil {
		return fmt.Errorf("updating document url: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("document %s: %w", id, ErrNotFound)
	}
	return nil
}

// GetDocumentByURL retrieves a document by its stored URL. Used by the
// read_document agent tool to resolve a URL the LLM saw in a previous
// citation back to a doc record. Exact match — the LLM passes URLs it
// copied verbatim from prior tool output, so no canonicalisation is
// needed; add it later if mismatches show up in logs. Empty URL never
// matches (NULL-URL docs are not addressable through this lookup).
func (s *SQLiteStore) GetDocumentByURL(ctx context.Context, url string) (*domain.Document, error) {
	if url == "" {
		return nil, fmt.Errorf("document with empty url: %w", ErrNotFound)
	}
	query := `
		SELECT id, title, content, url, language
		FROM documents
		WHERE url = ?
		LIMIT 1
	`
	var doc domain.Document
	err := s.db.QueryRowContext(ctx, query, url).Scan(
		&doc.ID, &doc.Title, &doc.Content, &doc.URL, &doc.Language,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("document with url %q: %w", url, ErrNotFound)
	}
	if err != nil {
		return nil, fmt.Errorf("querying document by url: %w", err)
	}
	return &doc, nil
}

// GetDocument retrieves a document by ID.
func (s *SQLiteStore) GetDocument(ctx context.Context, id string) (*domain.Document, error) {
	query := `
		SELECT id, title, content, url, language
		FROM documents
		WHERE id = ?
	`
	var doc domain.Document
	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&doc.ID, &doc.Title, &doc.Content, &doc.URL, &doc.Language,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("document %s: %w", id, ErrNotFound)
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
			SELECT id, title, content, url, language
			FROM documents
			WHERE language = ?
			ORDER BY created_at DESC
		`
		rows, err = s.db.QueryContext(ctx, query, *lang)
	} else {
		query := `
			SELECT id, title, content, url, language
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
			&doc.ID, &doc.Title, &doc.Content, &doc.URL, &doc.Language,
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

// DocumentSummary holds document metadata without full content.
//
// SourceKey is the source slug from crawl_pages.source_key. It is nil when
// the document was ingested via /v1/ingest (no crawl page) or when the
// crawl page row was deleted. When a document is reachable from multiple
// crawl_pages, SourceKey reflects the oldest one (the source that
// originally introduced the content).
type DocumentSummary struct {
	ID         string
	Title      string
	URL        *string
	Language   domain.Language
	SourceKey  *string
	ChunkCount int
	ContentLen int
}

// DocumentListFilter constrains ListDocumentsSummary results.
//
// Search filters by title/URL substring (LIKE). SourceKeys, when non-empty,
// restricts to documents linked to at least one crawl_page whose source_key is
// in the set; an empty/nil slice means no source filter.
type DocumentListFilter struct {
	Search     string
	SourceKeys []string
}

// CountDocuments returns the total number of documents.
func (s *SQLiteStore) CountDocuments(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM documents`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("counting documents: %w", err)
	}
	return count, nil
}

// CountChunks returns the total number of chunks.
func (s *SQLiteStore) CountChunks(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("counting chunks: %w", err)
	}
	return count, nil
}

// ListDocumentsSummary returns paginated document summaries (no content),
// optionally filtered by search text and/or source keys.
func (s *SQLiteStore) ListDocumentsSummary(ctx context.Context, limit, offset int, filter DocumentListFilter) ([]DocumentSummary, int, error) {
	whereClauses, whereArgs := buildDocumentFilterWhere(filter)

	countQuery := `SELECT COUNT(*) FROM documents d`
	if whereClauses != "" {
		countQuery += ` WHERE ` + whereClauses
	}
	var total int
	if err := s.db.QueryRowContext(ctx, countQuery, whereArgs...).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("counting documents: %w", err)
	}

	// Fetch page. SourceKey comes from the oldest crawl_page (ORDER BY cp.id
	// LIMIT 1) — when a doc is reached from multiple sources we surface the
	// one that originally introduced the content, rather than letting SQLite
	// pick whichever row the planner walks first (which would flicker across
	// requests).
	query := `
		SELECT d.id, d.title, d.url, d.language,
			LENGTH(d.content) as content_len,
			(SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) as chunk_count,
			(SELECT cp.source_key FROM crawl_pages cp WHERE cp.doc_id = d.id ORDER BY cp.id LIMIT 1) as source_key
		FROM documents d
	`
	args := append([]any(nil), whereArgs...)
	if whereClauses != "" {
		query += ` WHERE ` + whereClauses
	}
	query += ` ORDER BY d.created_at DESC LIMIT ? OFFSET ?`
	args = append(args, limit, offset)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("listing document summaries: %w", err)
	}
	defer rows.Close()

	var docs []DocumentSummary
	for rows.Next() {
		var d DocumentSummary
		var sourceKey sql.NullString
		if err := rows.Scan(&d.ID, &d.Title, &d.URL, &d.Language, &d.ContentLen, &d.ChunkCount, &sourceKey); err != nil {
			return nil, 0, fmt.Errorf("scanning document summary: %w", err)
		}
		if sourceKey.Valid {
			s := sourceKey.String
			d.SourceKey = &s
		}
		docs = append(docs, d)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterating document summaries: %w", err)
	}

	return docs, total, nil
}

// buildDocumentFilterWhere assembles the WHERE clause and bind args for the
// shared filter shape used by both the count and the page query in
// ListDocumentsSummary. Returns ("", nil) when no filter is active.
func buildDocumentFilterWhere(filter DocumentListFilter) (string, []any) {
	var clauses []string
	var args []any

	if filter.Search != "" {
		clauses = append(clauses, "(d.title LIKE ? OR d.url LIKE ?)")
		like := "%" + filter.Search + "%"
		args = append(args, like, like)
	}

	if len(filter.SourceKeys) > 0 {
		placeholders := make([]string, len(filter.SourceKeys))
		for i, k := range filter.SourceKeys {
			placeholders[i] = "?"
			args = append(args, k)
		}
		clauses = append(clauses,
			"EXISTS (SELECT 1 FROM crawl_pages cp WHERE cp.doc_id = d.id AND cp.source_key IN ("+strings.Join(placeholders, ",")+"))")
	}

	if len(clauses) == 0 {
		return "", nil
	}
	return strings.Join(clauses, " AND "), args
}

// DeleteDocument deletes a document and its chunks + embeddings (via CASCADE),
// then drops the matching entries from the in-memory cache.
func (s *SQLiteStore) DeleteDocument(ctx context.Context, id string) error {
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
		return fmt.Errorf("document %s: %w", id, ErrNotFound)
	}

	s.cache.removeByDocIDs([]string{id})
	return nil
}
