package store

import (
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite"
)

// OpenDB opens a SQLite database using the pure Go modernc.org/sqlite driver.
// Pragmas are set via DSN parameters so they apply to every pooled connection.
func OpenDB(path string) (*sql.DB, error) {
	dsn := path + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	return db, nil
}

// Migrate runs database migrations.
func Migrate(db *sql.DB, embeddingDimension int) error {
	schema := `
		CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL,
			content TEXT NOT NULL,
			url TEXT,
			language TEXT NOT NULL CHECK (language IN ('es', 'en')),
			category TEXT NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS chunks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			content TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			UNIQUE(doc_id, chunk_index)
		);

		CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

		CREATE TABLE IF NOT EXISTS chunk_embeddings (
			chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
			embedding BLOB NOT NULL
		);

		CREATE TABLE IF NOT EXISTS backends (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL UNIQUE,
			base_url TEXT NOT NULL,
			api_key TEXT NOT NULL DEFAULT '',
			enabled INTEGER NOT NULL DEFAULT 1,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS crawl_sources (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL,
			seed_url TEXT NOT NULL UNIQUE,
			category TEXT NOT NULL,
			language TEXT NOT NULL DEFAULT 'es',
			enabled INTEGER NOT NULL DEFAULT 1,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS crawl_pages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_id INTEGER NOT NULL REFERENCES crawl_sources(id),
			url TEXT NOT NULL UNIQUE,
			content_type TEXT NOT NULL DEFAULT 'html',
			title TEXT NOT NULL DEFAULT '',
			content_hash TEXT NOT NULL DEFAULT '',
			object_store_key TEXT NOT NULL DEFAULT '',
			doc_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
			status TEXT NOT NULL DEFAULT 'pending',
			error_message TEXT,
			last_crawled_at DATETIME,
			last_changed_at DATETIME,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE INDEX IF NOT EXISTS idx_crawl_pages_source ON crawl_pages(source_id);
		CREATE INDEX IF NOT EXISTS idx_crawl_pages_status ON crawl_pages(status);

		CREATE TABLE IF NOT EXISTS crawl_runs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_id INTEGER REFERENCES crawl_sources(id),
			status TEXT NOT NULL DEFAULT 'running',
			pages_discovered INTEGER NOT NULL DEFAULT 0,
			pages_fetched INTEGER NOT NULL DEFAULT 0,
			pages_changed INTEGER NOT NULL DEFAULT 0,
			pages_failed INTEGER NOT NULL DEFAULT 0,
			error_message TEXT,
			started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			finished_at DATETIME
		);
	`

	_, err := db.Exec(schema)
	if err != nil {
		return fmt.Errorf("running migrations: %w", err)
	}

	return nil
}
