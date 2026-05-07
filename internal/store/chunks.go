package store

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"

	"etsi-rag/internal/domain"
)

// serializeFloat32 converts a float32 slice to a little-endian byte slice.
func serializeFloat32(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// deserializeFloat32 converts a little-endian byte slice to a float32 slice.
func deserializeFloat32(buf []byte) []float32 {
	n := len(buf) / 4
	v := make([]float32, n)
	for i := range n {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return v
}

// cosineSimilarity computes cosine similarity between two vectors.
// Returns 0 if either vector has zero magnitude.
func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}

// CreateChunks inserts chunks with their embeddings, then mirrors the new
// entries into the in-memory cache (no-op when the cache is unloaded).
func (s *SQLiteStore) CreateChunks(ctx context.Context, docID string, chunks []domain.Chunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunks and embeddings count mismatch: %d vs %d", len(chunks), len(embeddings))
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	chunkStmt, err := tx.PrepareContext(ctx, `
		INSERT INTO chunks (doc_id, content, chunk_index, section_title)
		VALUES (?, ?, ?, ?)
	`)
	if err != nil {
		return fmt.Errorf("preparing chunk statement: %w", err)
	}
	defer chunkStmt.Close()

	embStmt, err := tx.PrepareContext(ctx, `
		INSERT INTO chunk_embeddings (chunk_id, embedding)
		VALUES (?, ?)
	`)
	if err != nil {
		return fmt.Errorf("preparing embedding statement: %w", err)
	}
	defer embStmt.Close()

	chunkIDs := make([]int64, len(chunks))
	for i, chunk := range chunks {
		if dim := s.embeddingDim(); len(embeddings[i]) != dim {
			return fmt.Errorf("embedding dimension mismatch at index %d: expected %d, got %d",
				i, dim, len(embeddings[i]))
		}

		result, err := chunkStmt.ExecContext(ctx, docID, chunk.Content, chunk.ChunkIndex, chunk.SectionTitle)
		if err != nil {
			return fmt.Errorf("inserting chunk %d: %w", i, err)
		}

		chunkID, err := result.LastInsertId()
		if err != nil {
			return fmt.Errorf("getting chunk ID: %w", err)
		}
		chunkIDs[i] = chunkID

		blob := serializeFloat32(embeddings[i])
		if _, err := embStmt.ExecContext(ctx, chunkID, blob); err != nil {
			return fmt.Errorf("inserting embedding %d: %w", i, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	entries := make([]cachedVec, len(chunks))
	for i := range chunks {
		entries[i] = cachedVec{
			ChunkID: chunkIDs[i],
			DocID:   docID,
			Vec:     embeddings[i],
		}
	}
	if err := s.cache.appendChunks(entries); err != nil {
		return fmt.Errorf("appending to cache: %w", err)
	}

	return nil
}

// GetChunksByDocument retrieves all chunks for a document.
func (s *SQLiteStore) GetChunksByDocument(ctx context.Context, docID string) ([]domain.Chunk, error) {
	query := `
		SELECT id, doc_id, content, chunk_index, section_title
		FROM chunks
		WHERE doc_id = ?
		ORDER BY chunk_index
	`
	rows, err := s.db.QueryContext(ctx, query, docID)
	if err != nil {
		return nil, fmt.Errorf("querying chunks: %w", err)
	}
	defer rows.Close()

	var chunks []domain.Chunk
	for rows.Next() {
		var chunk domain.Chunk
		if err := rows.Scan(&chunk.ID, &chunk.DocID, &chunk.Content, &chunk.ChunkIndex, &chunk.SectionTitle); err != nil {
			return nil, fmt.Errorf("scanning chunk: %w", err)
		}
		chunks = append(chunks, chunk)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating chunks: %w", err)
	}

	return chunks, nil
}

// SearchSimilar performs brute-force cosine similarity search. When the
// in-memory cache is loaded (Warm has run), scoring happens against
// pre-deserialized vectors and only the metadata for the top-K hits is
// fetched from SQLite. When the cache is unloaded, falls back to the full
// SQL scan path — keeping :memory: tests that never call Warm working.
//
// Optional sourceKeys narrow the search to chunks whose document is
// referenced by at least one crawl_pages row matching the given source key
// (the agent's list_sources / source_keys scoping path). Empty/no slice =
// search the entire corpus, identical to the pre-filter behaviour.
func (s *SQLiteStore) SearchSimilar(ctx context.Context, embedding []float32, topK int, sourceKeys ...string) ([]SearchResult, error) {
	if dim := s.embeddingDim(); len(embedding) != dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
			dim, len(embedding))
	}

	allowed, err := s.allowedDocIDsBySourceKeys(ctx, sourceKeys)
	if err != nil {
		return nil, err
	}
	if allowed != nil && len(allowed) == 0 {
		return nil, nil // sourceKeys provided but none of them have any pages
	}

	scored, useCache, err := s.cache.searchTopK(embedding, topK, allowed)
	if err != nil {
		return nil, fmt.Errorf("scoring against cache: %w", err)
	}
	if !useCache {
		return s.searchSimilarSQL(ctx, embedding, topK, allowed)
	}
	if len(scored) == 0 {
		return nil, nil
	}

	return s.fetchSearchResults(ctx, scored)
}

// allowedDocIDsBySourceKeys returns the set of doc IDs reachable via at least
// one crawl_pages row whose source_key is in the provided slice. nil sourceKeys
// → returns nil (the predicate "no filter"). Non-empty sourceKeys → returns a
// non-nil set (possibly empty if no pages match).
func (s *SQLiteStore) allowedDocIDsBySourceKeys(ctx context.Context, sourceKeys []string) (map[string]struct{}, error) {
	if len(sourceKeys) == 0 {
		return nil, nil
	}

	placeholders := make([]string, len(sourceKeys))
	args := make([]any, len(sourceKeys))
	for i, k := range sourceKeys {
		placeholders[i] = "?"
		args[i] = k
	}

	query := `SELECT DISTINCT doc_id FROM crawl_pages
		WHERE source_key IN (` + strings.Join(placeholders, ",") + `)
		AND doc_id IS NOT NULL`
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("resolving allowed doc IDs: %w", err)
	}
	defer rows.Close()

	allowed := make(map[string]struct{})
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scanning doc id: %w", err)
		}
		allowed[id] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating doc ids: %w", err)
	}
	return allowed, nil
}

// fetchSearchResults loads chunk + document metadata for the given top-K
// scored chunk IDs in a single batched query, preserving score-descending
// order.
func (s *SQLiteStore) fetchSearchResults(ctx context.Context, scored []cacheScored) ([]SearchResult, error) {
	scoreMap := make(map[int64]float32, len(scored))
	placeholders := make([]string, len(scored))
	fetchArgs := make([]any, len(scored))
	for i, sc := range scored {
		scoreMap[sc.ChunkID] = sc.Score
		placeholders[i] = "?"
		fetchArgs[i] = sc.ChunkID
	}

	fetchQuery := `
		SELECT
			c.id, c.doc_id, c.content, c.chunk_index, c.section_title,
			d.id, d.title, d.url, d.language, d.category
		FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		WHERE c.id IN (` + strings.Join(placeholders, ",") + `)
	`
	rows, err := s.db.QueryContext(ctx, fetchQuery, fetchArgs...)
	if err != nil {
		return nil, fmt.Errorf("fetching top-K chunks: %w", err)
	}
	defer rows.Close()

	results := make([]SearchResult, 0, len(scored))
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(
			&r.Chunk.ID, &r.Chunk.DocID, &r.Chunk.Content, &r.Chunk.ChunkIndex, &r.Chunk.SectionTitle,
			&r.Document.ID, &r.Document.Title, &r.Document.URL, &r.Document.Language, &r.Document.Category,
		); err != nil {
			return nil, fmt.Errorf("scanning result: %w", err)
		}
		r.Score = scoreMap[r.Chunk.ID]
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating results: %w", err)
	}

	// Re-sort by score descending — the IN query doesn't preserve order.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// sanitizeFTS5Query turns user-supplied free text into a safe FTS5 MATCH
// expression. Every non-alphanumeric Unicode rune is replaced with a space
// (so tokens like "unique-token-xyz" split the same way the unicode61
// tokenizer indexes chunk content), then the resulting tokens are joined
// with " OR " for recall-friendly matching — BM25 handles ranking. This
// also defuses any FTS5 operator characters (parens, quotes, *, :, ^, etc.)
// that the user might pass intentionally or otherwise. Returns "" when
// the cleaned input has no usable tokens; callers treat that as an empty
// result rather than an error.
func sanitizeFTS5Query(q string) string {
	var b strings.Builder
	b.Grow(len(q))
	for _, r := range q {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	tokens := strings.Fields(b.String())
	return strings.Join(tokens, " OR ")
}

// SearchKeyword runs a BM25 query against the chunks_fts virtual table.
// Returns the top-K results ordered by relevance (best first). The caller's
// query is split on whitespace and joined with " OR " — recall-friendly,
// BM25 handles ranking. Empty / whitespace-only input returns no results.
//
// Score on the returned results is the negated FTS5 bm25 value, so the
// "higher is better" invariant matches SearchSimilar and the rest of the
// codebase.
//
// FTS5's bm25() implements the BM25F generalisation; for a single-column
// table this reduces to standard BM25 with k1=1.2 and b=0.75 (Robertson's
// canonical defaults).
//
// Sources:
//
//	Robertson, S. E., et al. (1995). Okapi at TREC-3. NIST SP 500-225.
//	Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance
//	framework: BM25 and beyond. Foundations and Trends in IR, 3(4),
//	333–389. doi:10.1561/1500000019
//	SQLite FTS5 bm25() docs: https://www.sqlite.org/fts5.html#the_bm25_function
//
// See docs/retrieval.md for the formula and project-specific parameters.
func (s *SQLiteStore) SearchKeyword(ctx context.Context, query string, topK int, sourceKeys ...string) ([]SearchResult, error) {
	if topK <= 0 {
		return nil, nil
	}
	cleaned := sanitizeFTS5Query(query)
	if cleaned == "" {
		return nil, nil
	}

	allowed, err := s.allowedDocIDsBySourceKeys(ctx, sourceKeys)
	if err != nil {
		return nil, err
	}
	if allowed != nil && len(allowed) == 0 {
		return nil, nil
	}

	sql := `SELECT rowid, bm25(chunks_fts) AS score
		FROM chunks_fts
		WHERE chunks_fts MATCH ?`
	args := []any{cleaned}
	if allowed != nil {
		placeholders := make([]string, 0, len(allowed))
		for id := range allowed {
			placeholders = append(placeholders, "?")
			args = append(args, id)
		}
		sql += ` AND rowid IN (SELECT id FROM chunks WHERE doc_id IN (` +
			strings.Join(placeholders, ",") + `))`
	}
	sql += ` ORDER BY score LIMIT ?`
	args = append(args, topK)

	rows, err := s.db.QueryContext(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("bm25 search: %w", err)
	}
	defer rows.Close()

	var scored []cacheScored
	for rows.Next() {
		var chunkID int64
		var bm25Value float64
		if err := rows.Scan(&chunkID, &bm25Value); err != nil {
			return nil, fmt.Errorf("scanning bm25 row: %w", err)
		}
		// FTS5 bm25 returns lower = better. Negate so downstream code can
		// keep the "higher is better" invariant uniformly across retrievers.
		scored = append(scored, cacheScored{ChunkID: chunkID, Score: float32(-bm25Value)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating bm25 rows: %w", err)
	}

	if len(scored) == 0 {
		return nil, nil
	}
	return s.fetchSearchResults(ctx, scored)
}

// SearchInDocument performs a dense cosine search restricted to chunks of a
// single document. Powers the read_document tool's query mode: when the LLM
// asks "what does this doc say about X" we score only that doc's chunks.
//
// SQL-based (cache-bypass): the cache wasn't built for doc-scoped filtering
// and a single doc has at most a few dozen chunks, so the JOIN-and-score
// path is sub-millisecond. Mirrors SearchSimilar in semantics — unknown
// docID returns an empty slice (not an error) so the caller can map the
// "no hits" case to a graceful tool reply.
func (s *SQLiteStore) SearchInDocument(ctx context.Context, docID string, embedding []float32, topK int) ([]SearchResult, error) {
	if topK <= 0 {
		return nil, nil
	}
	if dim := s.embeddingDim(); len(embedding) != dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
			dim, len(embedding))
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT ce.chunk_id, ce.embedding
		FROM chunk_embeddings ce
		JOIN chunks c ON c.id = ce.chunk_id
		WHERE c.doc_id = ?
	`, docID)
	if err != nil {
		return nil, fmt.Errorf("loading doc chunk embeddings: %w", err)
	}
	defer rows.Close()

	var scored []cacheScored
	for rows.Next() {
		var chunkID int64
		var blob []byte
		if err := rows.Scan(&chunkID, &blob); err != nil {
			return nil, fmt.Errorf("scanning chunk embedding: %w", err)
		}
		vec := deserializeFloat32(blob)
		if dim := s.embeddingDim(); len(vec) != dim {
			return nil, fmt.Errorf("stored embedding dim mismatch for chunk %d: expected %d, got %d",
				chunkID, dim, len(vec))
		}
		scored = append(scored, cacheScored{ChunkID: chunkID, Score: cosineSimilarity(embedding, vec)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating doc chunk embeddings: %w", err)
	}

	if len(scored) == 0 {
		return nil, nil
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topK > len(scored) {
		topK = len(scored)
	}
	return s.fetchSearchResults(ctx, scored[:topK])
}

// searchSimilarSQL is the legacy full-scan path. Used when the cache hasn't
// been warmed (typical for :memory: tests). Loads every embedding row,
// deserializes, scores, sorts, then fetches metadata for the top-K.
//
// allowed (when non-nil) restricts scoring to chunks whose document is in the
// set — used by the source-key scoping path. The JOIN to chunks lets the
// SQL filter happen at the query level instead of post-load filtering.
func (s *SQLiteStore) searchSimilarSQL(ctx context.Context, embedding []float32, topK int, allowed map[string]struct{}) ([]SearchResult, error) {
	query := `SELECT ce.chunk_id, ce.embedding FROM chunk_embeddings ce`
	var args []any
	if allowed != nil {
		placeholders := make([]string, 0, len(allowed))
		args = make([]any, 0, len(allowed))
		for id := range allowed {
			placeholders = append(placeholders, "?")
			args = append(args, id)
		}
		query += ` JOIN chunks c ON c.id = ce.chunk_id WHERE c.doc_id IN (` +
			strings.Join(placeholders, ",") + `)`
	}
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("loading embeddings: %w", err)
	}
	defer rows.Close()

	var scored []cacheScored
	for rows.Next() {
		var chunkID int64
		var blob []byte
		if err := rows.Scan(&chunkID, &blob); err != nil {
			return nil, fmt.Errorf("scanning embedding: %w", err)
		}
		vec := deserializeFloat32(blob)
		if dim := s.embeddingDim(); len(vec) != dim {
			return nil, fmt.Errorf("stored embedding dimension mismatch for chunk %d: expected %d, got %d",
				chunkID, dim, len(vec))
		}
		scored = append(scored, cacheScored{ChunkID: chunkID, Score: cosineSimilarity(embedding, vec)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating embeddings: %w", err)
	}

	if len(scored) == 0 {
		return nil, nil
	}

	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topK > len(scored) {
		topK = len(scored)
	}
	return s.fetchSearchResults(ctx, scored[:topK])
}
