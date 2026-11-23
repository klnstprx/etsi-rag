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
			search_prefix TEXT NOT NULL DEFAULT '',
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

		-- FTS5 keyword index over search_prefix, section_title and content
		-- (see ftsTriggers). The chunker stores plain body in chunks.content
		-- and the heading breadcrumb in chunks.section_title; the triggers
		-- concatenate them so BM25 sees title and heading words alongside
		-- the body. External-content
		-- table (content='chunks', content_rowid='id') so the index
		-- references rows in the chunks table without duplicating bytes;
		-- snippet()/highlight() are not used (only bm25()), so the
		-- index-vs-content asymmetry is fine. unicode61 with
		-- remove_diacritics=2 folds Spanish accents (María ↔ maria).
		CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
			content,
			content='chunks',
			content_rowid='id',
			tokenize='unicode61 remove_diacritics 2'
		);

		-- Chunks created before the chunks_fts table existed are not
		-- retroactively indexed. To populate the FTS5 index over a
		-- pre-existing corpus, re-crawl the affected sources.
	`

	if _, err := db.Exec(schema); err != nil {
		return fmt.Errorf("running migrations: %w", err)
	}
	if err := addColumn(db, "chunks", "search_prefix", "TEXT NOT NULL DEFAULT ''"); err != nil {
		return err
	}
	// Triggers keep chunks_fts in sync with chunks. The indexed value is
	// search_prefix (the document title, when chunking.title_in_search_text
	// is on), section_title and content, so title and heading words count in
	// BM25 scoring. They are recreated on every start so an older database
	// picks up the current expression; rows indexed before search_prefix
	// existed have it empty, which tokenises the same as the old expression.
	if _, err := db.Exec(ftsTriggers); err != nil {
		return fmt.Errorf("creating chunks_fts triggers: %w", err)
	}
	return nil
}

const ftsIndexed = `coalesce(%[1]s.search_prefix, '') || ' ' || coalesce(%[1]s.section_title, '') || ' ' || %[1]s.content`

var ftsTriggers = fmt.Sprintf(`
	DROP TRIGGER IF EXISTS chunks_ai;
	DROP TRIGGER IF EXISTS chunks_ad;
	DROP TRIGGER IF EXISTS chunks_au;
	CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
		INSERT INTO chunks_fts(rowid, content) VALUES (new.id, %[1]s);
	END;
	CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
		INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, %[2]s);
	END;
	CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
		INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, %[2]s);
		INSERT INTO chunks_fts(rowid, content) VALUES (new.id, %[1]s);
	END;`, fmt.Sprintf(ftsIndexed, "new"), fmt.Sprintf(ftsIndexed, "old"))

// addColumn adds a column to an existing table unless it is already there.
func addColumn(db *sql.DB, table, column, definition string) error {
	rows, err := db.Query(`SELECT name FROM pragma_table_info(?)`, table)
	if err != nil {
		return fmt.Errorf("reading %s columns: %w", table, err)
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return fmt.Errorf("reading %s columns: %w", table, err)
		}
		if name == column {
			return nil
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("reading %s columns: %w", table, err)
	}
	if _, err := db.Exec(fmt.Sprintf(`ALTER TABLE %s ADD COLUMN %s %s`, table, column, definition)); err != nil {
		return fmt.Errorf("adding %s.%s: %w", table, column, err)
	}
	return nil
}
