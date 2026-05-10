package store

import (
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite"
)

// OpenDB opens a SQLite database using the pure Go modernc.org/sqlite driver.
// Pragmas are set via DSN parameters so they apply to every pooled connection.
func OpenDB(path string) (*sql.DB, error) {
	dsn := path + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)&_pragma=cache_size(-4000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	return db, nil
}

// Migrate runs database migrations.
func Migrate(db *sql.DB) error {
	schema := `
		CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL,
			content TEXT NOT NULL,
			url TEXT,
			language TEXT NOT NULL CHECK (language IN ('es', 'en')),
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS chunks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			content TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			section_title TEXT NOT NULL DEFAULT '',
			image_refs TEXT NOT NULL DEFAULT '',
			UNIQUE(doc_id, chunk_index)
		);

		CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
		CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

		CREATE TABLE IF NOT EXISTS chunk_embeddings (
			chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
			embedding BLOB NOT NULL
		);

		-- crawl_sources lives in config/sources.toml. crawl_pages and crawl_runs
		-- reference it by stable slug (source_key) so renaming a seed URL or
		-- display name in the TOML does not orphan history. The slug is
		-- validated against the loaded sources at runtime, not by SQL FK.
		CREATE TABLE IF NOT EXISTS crawl_pages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_key TEXT NOT NULL,
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

		CREATE INDEX IF NOT EXISTS idx_crawl_pages_source ON crawl_pages(source_key);
		CREATE INDEX IF NOT EXISTS idx_crawl_pages_status ON crawl_pages(status);
		CREATE INDEX IF NOT EXISTS idx_crawl_pages_doc_id ON crawl_pages(doc_id);

		CREATE TABLE IF NOT EXISTS crawl_runs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_key TEXT,
			status TEXT NOT NULL DEFAULT 'running',
			pages_discovered INTEGER NOT NULL DEFAULT 0,
			pages_fetched INTEGER NOT NULL DEFAULT 0,
			pages_changed INTEGER NOT NULL DEFAULT 0,
			pages_failed INTEGER NOT NULL DEFAULT 0,
			error_message TEXT,
			started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			finished_at DATETIME
		);

		CREATE TABLE IF NOT EXISTS document_artifacts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			ref TEXT NOT NULL,
			kind TEXT NOT NULL,
			object_store_key TEXT NOT NULL,
			mime_type TEXT NOT NULL,
			width INTEGER NOT NULL DEFAULT 0,
			height INTEGER NOT NULL DEFAULT 0,
			public_token TEXT NOT NULL UNIQUE,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(doc_id, ref)
		);

		CREATE INDEX IF NOT EXISTS idx_artifacts_doc ON document_artifacts(doc_id);

		-- FTS5 keyword index over chunks.content. External-content table
		-- (content='chunks', content_rowid='id') so the index references rows
		-- in the chunks table without duplicating bytes. unicode61 with
		-- remove_diacritics=2 folds Spanish accents (María ↔ maria).
		CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
			content,
			content='chunks',
			content_rowid='id',
			tokenize='unicode61 remove_diacritics 2'
		);

		-- Triggers keep chunks_fts in sync with chunks.
		CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
			INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
		END;
		CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
			INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
		END;
		CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
			INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
			INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
		END;

		-- Chunks created before the chunks_fts table existed are not
		-- retroactively indexed. To populate the FTS5 index over a
		-- pre-existing corpus, re-crawl the affected sources.
	`

	_, err := db.Exec(schema)
	if err != nil {
		return fmt.Errorf("running migrations: %w", err)
	}

	return nil
}
