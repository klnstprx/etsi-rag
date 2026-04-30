package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"etsi-rag/internal/domain"
)

// --- Sources ---

// ListCrawlSources returns all crawl sources ordered by ID.
func (s *SQLiteStore) ListCrawlSources(ctx context.Context) ([]domain.CrawlSource, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, name, seed_url, category, language, enabled
		FROM crawl_sources ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("listing crawl sources: %w", err)
	}
	defer rows.Close()

	var sources []domain.CrawlSource
	for rows.Next() {
		var src domain.CrawlSource
		if err := rows.Scan(&src.ID, &src.Name, &src.SeedURL, &src.Category,
			&src.Language, &src.Enabled); err != nil {
			return nil, fmt.Errorf("scanning crawl source: %w", err)
		}
		sources = append(sources, src)
	}
	return sources, rows.Err()
}

// CreateCrawlSource inserts a new crawl source (INSERT OR IGNORE for idempotent seeding).
func (s *SQLiteStore) CreateCrawlSource(ctx context.Context, src *domain.CrawlSource) error {
	result, err := s.db.ExecContext(ctx, `
		INSERT OR IGNORE INTO crawl_sources (name, seed_url, category, language, enabled)
		VALUES (?, ?, ?, ?, ?)`,
		src.Name, src.SeedURL, src.Category, src.Language, src.Enabled)
	if err != nil {
		return fmt.Errorf("creating crawl source: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("getting crawl source id: %w", err)
	}
	if id != 0 {
		src.ID = id
	}
	return nil
}

// UpdateCrawlSource updates an existing crawl source.
func (s *SQLiteStore) UpdateCrawlSource(ctx context.Context, src *domain.CrawlSource) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE crawl_sources SET name = ?, seed_url = ?, category = ?,
			language = ?, enabled = ?
		WHERE id = ?`,
		src.Name, src.SeedURL, src.Category, src.Language, src.Enabled, src.ID)
	if err != nil {
		return fmt.Errorf("updating crawl source: %w", err)
	}
	return nil
}

// DeleteCrawlSource removes a crawl source by ID.
func (s *SQLiteStore) DeleteCrawlSource(ctx context.Context, id int64) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM crawl_sources WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("deleting crawl source: %w", err)
	}
	return nil
}

// --- Pages ---

// UpsertCrawlPage inserts or updates a crawl page by URL.
func (s *SQLiteStore) UpsertCrawlPage(ctx context.Context, page *domain.CrawlPage) error {
	result, err := s.db.ExecContext(ctx, `
		INSERT INTO crawl_pages (source_id, url, content_type, title, content_hash,
			object_store_key, doc_id, status, error_message, last_crawled_at, last_changed_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(url) DO UPDATE SET
			content_type = excluded.content_type,
			title = excluded.title,
			content_hash = excluded.content_hash,
			object_store_key = excluded.object_store_key,
			doc_id = excluded.doc_id,
			status = excluded.status,
			error_message = excluded.error_message,
			last_crawled_at = excluded.last_crawled_at,
			last_changed_at = excluded.last_changed_at`,
		page.SourceID, page.URL, page.ContentType, page.Title, page.ContentHash,
		page.ObjectStoreKey, page.DocID, string(page.Status), page.ErrorMessage,
		page.LastCrawledAt, page.LastChangedAt)
	if err != nil {
		return fmt.Errorf("upserting crawl page: %w", err)
	}
	id, _ := result.LastInsertId()
	if id != 0 {
		page.ID = id
	}
	return nil
}

// GetCrawlPageByURL returns a crawl page by its URL.
func (s *SQLiteStore) GetCrawlPageByURL(ctx context.Context, url string) (*domain.CrawlPage, error) {
	var page domain.CrawlPage
	var status string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, source_id, url, content_type, title, content_hash,
			object_store_key, doc_id, status, error_message, last_crawled_at, last_changed_at
		FROM crawl_pages WHERE url = ?`, url).
		Scan(&page.ID, &page.SourceID, &page.URL, &page.ContentType, &page.Title,
			&page.ContentHash, &page.ObjectStoreKey, &page.DocID, &status,
			&page.ErrorMessage, &page.LastCrawledAt, &page.LastChangedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting crawl page: %w", err)
	}
	page.Status = domain.CrawlPageStatus(status)
	return &page, nil
}

// ListCrawlPages returns crawl pages, optionally filtered by source and/or status.
func (s *SQLiteStore) ListCrawlPages(ctx context.Context, sourceID *int64, status *string) ([]domain.CrawlPage, error) {
	query := `SELECT id, source_id, url, content_type, title, content_hash,
		object_store_key, doc_id, status, error_message, last_crawled_at, last_changed_at
		FROM crawl_pages WHERE 1=1`
	var args []any

	if sourceID != nil {
		query += ` AND source_id = ?`
		args = append(args, *sourceID)
	}
	if status != nil {
		query += ` AND status = ?`
		args = append(args, *status)
	}
	query += ` ORDER BY id`

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("listing crawl pages: %w", err)
	}
	defer rows.Close()

	var pages []domain.CrawlPage
	for rows.Next() {
		var p domain.CrawlPage
		var st string
		if err := rows.Scan(&p.ID, &p.SourceID, &p.URL, &p.ContentType, &p.Title,
			&p.ContentHash, &p.ObjectStoreKey, &p.DocID, &st,
			&p.ErrorMessage, &p.LastCrawledAt, &p.LastChangedAt); err != nil {
			return nil, fmt.Errorf("scanning crawl page: %w", err)
		}
		p.Status = domain.CrawlPageStatus(st)
		pages = append(pages, p)
	}
	return pages, rows.Err()
}

// MarkStalePages marks pages not crawled since runStartedAt as stale.
func (s *SQLiteStore) MarkStalePages(ctx context.Context, sourceID int64, runStartedAt time.Time) (int, error) {
	result, err := s.db.ExecContext(ctx, `
		UPDATE crawl_pages SET status = 'stale'
		WHERE source_id = ? AND status != 'stale'
			AND (last_crawled_at IS NULL OR last_crawled_at < ?)`,
		sourceID, runStartedAt)
	if err != nil {
		return 0, fmt.Errorf("marking stale pages: %w", err)
	}
	n, _ := result.RowsAffected()
	return int(n), nil
}

// PurgeOrphanDocuments deletes documents that are not referenced by any
// crawl_page. Orphans arise when the URL that pointed at a document moves to
// new content (content-hash-derived doc IDs mean the old doc stays behind
// until another URL also points away from it). Chunks + embeddings cascade.
//
// The target doc IDs are captured inside the same transaction as the DELETE
// so the cache invalidation hook drops exactly the rows that were removed —
// avoids races against concurrent writers between SELECT and DELETE.
func (s *SQLiteStore) PurgeOrphanDocuments(ctx context.Context) (int, error) {
	const cond = `WHERE id NOT IN (SELECT doc_id FROM crawl_pages WHERE doc_id IS NOT NULL)`
	purgedIDs, n, err := s.purgeDocumentsTx(ctx, "SELECT id FROM documents "+cond, "DELETE FROM documents "+cond)
	if err != nil {
		return 0, fmt.Errorf("purging orphan documents: %w", err)
	}
	s.cache.removeByDocIDs(purgedIDs)
	return n, nil
}

// PurgeStaleDocuments deletes documents linked to stale crawl pages.
func (s *SQLiteStore) PurgeStaleDocuments(ctx context.Context) (int, error) {
	const cond = `WHERE id IN (SELECT doc_id FROM crawl_pages WHERE status = 'stale' AND doc_id IS NOT NULL)`
	purgedIDs, n, err := s.purgeDocumentsTx(ctx, "SELECT id FROM documents "+cond, "DELETE FROM documents "+cond)
	if err != nil {
		return 0, fmt.Errorf("purging stale documents: %w", err)
	}
	s.cache.removeByDocIDs(purgedIDs)

	// Clear doc_id references on now-orphaned stale pages (best effort).
	s.db.ExecContext(ctx, `UPDATE crawl_pages SET doc_id = NULL WHERE status = 'stale'`)

	return n, nil
}

// purgeDocumentsTx runs a SELECT + DELETE pair against documents in the same
// transaction and returns the IDs that were deleted plus the affected count.
func (s *SQLiteStore) purgeDocumentsTx(ctx context.Context, selectQuery, deleteQuery string) ([]string, int, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, 0, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	rows, err := tx.QueryContext(ctx, selectQuery)
	if err != nil {
		return nil, 0, fmt.Errorf("collecting target doc IDs: %w", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, 0, fmt.Errorf("scanning target id: %w", err)
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterating target ids: %w", err)
	}

	result, err := tx.ExecContext(ctx, deleteQuery)
	if err != nil {
		return nil, 0, fmt.Errorf("deleting documents: %w", err)
	}
	n, _ := result.RowsAffected()

	if err := tx.Commit(); err != nil {
		return nil, 0, fmt.Errorf("committing transaction: %w", err)
	}
	return ids, int(n), nil
}

// --- Runs ---

// CreateCrawlRun creates a new crawl run and returns its ID.
func (s *SQLiteStore) CreateCrawlRun(ctx context.Context, sourceID *int64) (int64, error) {
	result, err := s.db.ExecContext(ctx, `
		INSERT INTO crawl_runs (source_id, status) VALUES (?, 'running')`, sourceID)
	if err != nil {
		return 0, fmt.Errorf("creating crawl run: %w", err)
	}
	return result.LastInsertId()
}

// UpdateCrawlRun updates a crawl run's counters and status.
func (s *SQLiteStore) UpdateCrawlRun(ctx context.Context, run *domain.CrawlRun) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE crawl_runs SET status = ?, pages_discovered = ?, pages_fetched = ?,
			pages_changed = ?, pages_failed = ?, error_message = ?, finished_at = ?
		WHERE id = ?`,
		string(run.Status), run.PagesDiscovered, run.PagesFetched,
		run.PagesChanged, run.PagesFailed, run.ErrorMessage, run.FinishedAt, run.ID)
	if err != nil {
		return fmt.Errorf("updating crawl run: %w", err)
	}
	return nil
}

// GetCrawlRun returns a crawl run by ID.
func (s *SQLiteStore) GetCrawlRun(ctx context.Context, id int64) (*domain.CrawlRun, error) {
	var run domain.CrawlRun
	var status string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, source_id, status, pages_discovered, pages_fetched,
			pages_changed, pages_failed, error_message, started_at, finished_at
		FROM crawl_runs WHERE id = ?`, id).
		Scan(&run.ID, &run.SourceID, &status, &run.PagesDiscovered, &run.PagesFetched,
			&run.PagesChanged, &run.PagesFailed, &run.ErrorMessage,
			&run.StartedAt, &run.FinishedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("crawl run %d: %w", id, ErrNotFound)
	}
	if err != nil {
		return nil, fmt.Errorf("getting crawl run: %w", err)
	}
	run.Status = domain.CrawlRunStatus(status)
	return &run, nil
}

// ListCrawlRuns returns the most recent crawl runs.
func (s *SQLiteStore) ListCrawlRuns(ctx context.Context, limit int) ([]domain.CrawlRun, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, source_id, status, pages_discovered, pages_fetched,
			pages_changed, pages_failed, error_message, started_at, finished_at
		FROM crawl_runs ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("listing crawl runs: %w", err)
	}
	defer rows.Close()

	var runs []domain.CrawlRun
	for rows.Next() {
		var run domain.CrawlRun
		var status string
		if err := rows.Scan(&run.ID, &run.SourceID, &status, &run.PagesDiscovered,
			&run.PagesFetched, &run.PagesChanged, &run.PagesFailed,
			&run.ErrorMessage, &run.StartedAt, &run.FinishedAt); err != nil {
			return nil, fmt.Errorf("scanning crawl run: %w", err)
		}
		run.Status = domain.CrawlRunStatus(status)
		runs = append(runs, run)
	}
	return runs, rows.Err()
}
