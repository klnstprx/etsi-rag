package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"etsi-rag/internal/domain"
)

// CrawlStore provides database operations for crawl-related entities.
type CrawlStore interface {
	// Sources
	ListCrawlSources(ctx context.Context) ([]domain.CrawlSource, error)
	CreateCrawlSource(ctx context.Context, src *domain.CrawlSource) error
	UpdateCrawlSource(ctx context.Context, src *domain.CrawlSource) error
	DeleteCrawlSource(ctx context.Context, id int64) error

	// Pages
	UpsertCrawlPage(ctx context.Context, page *domain.CrawlPage) error
	GetCrawlPageByURL(ctx context.Context, url string) (*domain.CrawlPage, error)
	ListCrawlPages(ctx context.Context, sourceID *int64, status *string) ([]domain.CrawlPage, error)
	MarkStalePages(ctx context.Context, sourceID int64, runStartedAt time.Time) (int, error)
	PurgeStaleDocuments(ctx context.Context) (int, error)

	// Runs
	CreateCrawlRun(ctx context.Context, sourceID *int64) (int64, error)
	UpdateCrawlRun(ctx context.Context, run *domain.CrawlRun) error
	GetCrawlRun(ctx context.Context, id int64) (*domain.CrawlRun, error)
	ListCrawlRuns(ctx context.Context, limit int) ([]domain.CrawlRun, error)
}

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

// PurgeStaleDocuments deletes documents linked to stale crawl pages.
func (s *SQLiteStore) PurgeStaleDocuments(ctx context.Context) (int, error) {
	result, err := s.db.ExecContext(ctx, `
		DELETE FROM documents WHERE id IN (
			SELECT doc_id FROM crawl_pages WHERE status = 'stale' AND doc_id IS NOT NULL
		)`)
	if err != nil {
		return 0, fmt.Errorf("purging stale documents: %w", err)
	}
	n, _ := result.RowsAffected()

	// Clear doc_id references on stale pages
	s.db.ExecContext(ctx, `UPDATE crawl_pages SET doc_id = NULL WHERE status = 'stale'`)

	return int(n), nil
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
		return nil, fmt.Errorf("crawl run not found: %d", id)
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
