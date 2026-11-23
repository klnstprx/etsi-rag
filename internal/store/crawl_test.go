package store

import (
	"context"
	"testing"
	"time"

	"etsi-rag/internal/domain"
)

func TestSourceKeyByDocID(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	if err := s.CreateDocument(ctx, &domain.Document{
		ID: "doc-1", Title: "T", Content: "c", Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("create doc: %v", err)
	}

	t.Run("doc with no crawl_page returns empty", func(t *testing.T) {
		key, err := s.SourceKeyByDocID(ctx, "doc-1")
		if err != nil {
			t.Fatalf("SourceKeyByDocID: %v", err)
		}
		if key != "" {
			t.Errorf("key = %q, want empty", key)
		}
	})

	t.Run("unknown doc id returns empty", func(t *testing.T) {
		key, err := s.SourceKeyByDocID(ctx, "doc-does-not-exist")
		if err != nil {
			t.Fatalf("SourceKeyByDocID: %v", err)
		}
		if key != "" {
			t.Errorf("key = %q, want empty", key)
		}
	})

	t.Run("single crawl_page returns its source_key", func(t *testing.T) {
		now := time.Now()
		docID := "doc-1"
		if err := s.UpsertCrawlPage(ctx, &domain.CrawlPage{
			SourceKey:     "alpha",
			URL:           "https://example.com/a",
			DocID:         &docID,
			ContentHash:   "h1",
			Status:        domain.CrawlPageFetched,
			LastCrawledAt: &now,
		}); err != nil {
			t.Fatalf("upsert: %v", err)
		}
		key, err := s.SourceKeyByDocID(ctx, "doc-1")
		if err != nil {
			t.Fatalf("SourceKeyByDocID: %v", err)
		}
		if key != "alpha" {
			t.Errorf("key = %q, want %q", key, "alpha")
		}
	})
}

// TestSourceKeyByDocID_PicksOldest verifies the ORDER BY id LIMIT 1
// guarantee: when a doc is reached from multiple sources, the call returns
// the source that was inserted first (lowest crawl_pages.id), and the result
// is stable across repeated calls.
func TestSourceKeyByDocID_PicksOldest(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	docID := "doc-multi"
	if err := s.CreateDocument(ctx, &domain.Document{
		ID: docID, Title: "T", Content: "c", Language: domain.LanguageES,
	}); err != nil {
		t.Fatalf("create doc: %v", err)
	}

	// Insert in a deliberate order: "second-source" first, then "first-source".
	// The call should still return "second-source" because it has the smaller
	// crawl_pages.id (autoincrement). The names are intentionally counter to
	// alphabetical / source_key ordering so the test would fail if the impl
	// silently changed to ORDER BY source_key.
	now := time.Now()
	pages := []*domain.CrawlPage{
		{SourceKey: "second-source", URL: "https://example.com/first-inserted", DocID: &docID, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "first-source", URL: "https://example.com/second-inserted", DocID: &docID, ContentHash: "h2", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
	}
	for _, p := range pages {
		if err := s.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upsert %s: %v", p.URL, err)
		}
	}

	for i := range 3 {
		key, err := s.SourceKeyByDocID(ctx, docID)
		if err != nil {
			t.Fatalf("SourceKeyByDocID (call %d): %v", i, err)
		}
		if key != "second-source" {
			t.Errorf("call %d: key = %q, want %q (ORDER BY id LIMIT 1 should pick the oldest row)", i, key, "second-source")
		}
	}
}

// TestPurgeStaleDocuments_KeepsDocumentsOfLivePages checks that a document a
// live page shares with a stale one through content dedup survives the purge,
// while a document only stale pages reference goes.
func TestPurgeStaleDocuments_KeepsDocumentsOfLivePages(t *testing.T) {
	s := setupTestStore(t)
	ctx := context.Background()

	shared, staleOnly := "crawl-shared", "crawl-stale"
	for _, id := range []string{shared, staleOnly} {
		if err := s.CreateDocument(ctx, &domain.Document{ID: id, Title: id, Content: "c", Language: domain.LanguageES}); err != nil {
			t.Fatalf("create doc: %v", err)
		}
	}
	now := time.Now()
	for _, p := range []*domain.CrawlPage{
		{SourceKey: "a", URL: "https://example.com/live", DocID: &shared, ContentHash: "h1", Status: domain.CrawlPageFetched, LastCrawledAt: &now},
		{SourceKey: "a", URL: "https://example.com/gone-twin", DocID: &shared, ContentHash: "h1", Status: domain.CrawlPageStale, LastCrawledAt: &now},
		{SourceKey: "a", URL: "https://example.com/gone", DocID: &staleOnly, ContentHash: "h2", Status: domain.CrawlPageStale, LastCrawledAt: &now},
	} {
		if err := s.UpsertCrawlPage(ctx, p); err != nil {
			t.Fatalf("upsert %s: %v", p.URL, err)
		}
	}

	n, err := s.PurgeStaleDocuments(ctx)
	if err != nil {
		t.Fatalf("PurgeStaleDocuments: %v", err)
	}
	if n != 1 {
		t.Errorf("purged %d documents, want 1", n)
	}
	if ok, _ := s.DocumentExists(ctx, shared); !ok {
		t.Error("document of a live page was purged")
	}
	if ok, _ := s.DocumentExists(ctx, staleOnly); ok {
		t.Error("document referenced only by stale pages survived")
	}
}
