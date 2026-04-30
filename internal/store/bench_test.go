package store

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkSearchSimilarRealDB measures brute-force cosine search latency
// against the actual crawled SQLite DB. Skips if the DB is not present.
//
// Override the path with ETSI_BENCH_DB. Default is ../../data/etsi-rag.db,
// which resolves correctly when running `go test ./internal/store/...`.
//
// Run with:
//
//	go test -bench=BenchmarkSearchSimilarRealDB -benchmem -run='^$' ./internal/store/
func BenchmarkSearchSimilarRealDB(b *testing.B) {
	path := os.Getenv("ETSI_BENCH_DB")
	if path == "" {
		path = filepath.Join("..", "..", "data", "etsi-rag.db")
	}
	if _, err := os.Stat(path); err != nil {
		b.Skipf("real DB not available at %s: %v", path, err)
	}

	dsn := path + "?mode=ro&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		b.Fatalf("opening DB: %v", err)
	}
	b.Cleanup(func() { db.Close() })

	ctx := context.Background()

	var queryBlob []byte
	if err := db.QueryRowContext(ctx,
		"SELECT embedding FROM chunk_embeddings ORDER BY chunk_id LIMIT 1",
	).Scan(&queryBlob); err != nil {
		b.Fatalf("loading sample embedding: %v", err)
	}
	dim := len(queryBlob) / 4
	query := deserializeFloat32(queryBlob)

	var totalChunks int
	if err := db.QueryRowContext(ctx, "SELECT COUNT(*) FROM chunk_embeddings").Scan(&totalChunks); err != nil {
		b.Fatalf("counting chunks: %v", err)
	}
	b.Logf("dim=%d total_chunks=%d", dim, totalChunks)

	store := NewSQLiteStore(db, func() int { return dim })

	b.Run("topK=10/uncached", func(b *testing.B) {
		b.ReportMetric(float64(totalChunks), "chunks")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := store.SearchSimilar(ctx, query, 10); err != nil {
				b.Fatalf("search: %v", err)
			}
		}
	})

	b.Run("topK=50/uncached", func(b *testing.B) {
		b.ReportMetric(float64(totalChunks), "chunks")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := store.SearchSimilar(ctx, query, 50); err != nil {
				b.Fatalf("search: %v", err)
			}
		}
	})

	b.Run("topK=10/cached", func(b *testing.B) {
		if _, err := store.Warm(ctx); err != nil {
			b.Fatalf("warming cache: %v", err)
		}
		b.ReportMetric(float64(totalChunks), "chunks")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := store.SearchSimilar(ctx, query, 10); err != nil {
				b.Fatalf("search: %v", err)
			}
		}
	})

	// In-memory variant: isolates pure cosine compute + sort cost from
	// the SQLite load + deserialize cost that dominates the uncached path.
	// Kept as a reference for the theoretical floor.
	b.Run("inmem_cosine_only", func(b *testing.B) {
		rows, err := db.QueryContext(ctx, `SELECT embedding FROM chunk_embeddings`)
		if err != nil {
			b.Fatalf("loading embeddings: %v", err)
		}
		vecs := make([][]float32, 0, totalChunks)
		for rows.Next() {
			var blob []byte
			if err := rows.Scan(&blob); err != nil {
				rows.Close()
				b.Fatalf("scan: %v", err)
			}
			vecs = append(vecs, deserializeFloat32(blob))
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			b.Fatalf("rows: %v", err)
		}
		b.ReportMetric(float64(len(vecs)), "chunks")

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			scores := make([]float32, len(vecs))
			for j, v := range vecs {
				scores[j] = cosineSimilarity(query, v)
			}
			idx := make([]int, len(scores))
			for j := range idx {
				idx[j] = j
			}
			topK := 10
			for k := range topK {
				best := k
				for j := k + 1; j < len(idx); j++ {
					if scores[idx[j]] > scores[idx[best]] {
						best = j
					}
				}
				idx[k], idx[best] = idx[best], idx[k]
			}
		}
	})
}
