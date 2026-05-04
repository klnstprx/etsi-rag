package rag

import (
	"math"
	"testing"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
)

// makeResult constructs a minimal SearchResult for ordering tests. ID is the
// only field that matters for fusion identity; Score is incidental — RRF
// ignores absolute scores and uses positional rank.
func makeResult(id int64, docID string) store.SearchResult {
	return store.SearchResult{
		Chunk:    domain.Chunk{ID: id, DocID: docID, Content: "c"},
		Document: domain.Document{ID: docID, Title: "t"},
		Score:    1.0,
	}
}

func ids(results []store.SearchResult) []int64 {
	out := make([]int64, len(results))
	for i, r := range results {
		out[i] = r.Chunk.ID
	}
	return out
}

func TestFuseRRF_BothEmpty(t *testing.T) {
	got := FuseRRF(nil, nil, 10, 60)
	if len(got) != 0 {
		t.Errorf("expected empty, got %d results", len(got))
	}
}

func TestFuseRRF_OneEmpty(t *testing.T) {
	dense := []store.SearchResult{makeResult(1, "d"), makeResult(2, "d")}
	got := FuseRRF(dense, nil, 10, 60)
	if want := []int64{1, 2}; !equalIDs(ids(got), want) {
		t.Errorf("dense-only: ids=%v want=%v", ids(got), want)
	}

	sparse := []store.SearchResult{makeResult(3, "d"), makeResult(4, "d")}
	got = FuseRRF(nil, sparse, 10, 60)
	if want := []int64{3, 4}; !equalIDs(ids(got), want) {
		t.Errorf("sparse-only: ids=%v want=%v", ids(got), want)
	}
}

func TestFuseRRF_DisjointLists(t *testing.T) {
	// Dense ranks A>B; sparse ranks C>D. Result should interleave by RRF
	// score: rank-1 in either list scores 1/61; rank-2 scores 1/62. So the
	// expected order is one of [A,C,B,D] or [C,A,D,B] — A and C tie at 1/61.
	dense := []store.SearchResult{makeResult(1, "d1"), makeResult(2, "d1")}
	sparse := []store.SearchResult{makeResult(3, "d2"), makeResult(4, "d2")}
	got := FuseRRF(dense, sparse, 10, 60)
	if len(got) != 4 {
		t.Fatalf("expected 4 results, got %d", len(got))
	}
	// Top 2 must be {1,3}; bottom 2 must be {2,4}, regardless of tie ordering.
	top := map[int64]bool{got[0].Chunk.ID: true, got[1].Chunk.ID: true}
	if !top[1] || !top[3] {
		t.Errorf("top-2 should be {1,3} (rank-1 in each list), got %v", ids(got)[:2])
	}
	bottom := map[int64]bool{got[2].Chunk.ID: true, got[3].Chunk.ID: true}
	if !bottom[2] || !bottom[4] {
		t.Errorf("bottom-2 should be {2,4}, got %v", ids(got)[2:])
	}
}

func TestFuseRRF_ChunkInBothListsScoresHigher(t *testing.T) {
	// Chunk #1 is rank-1 in dense AND rank-1 in sparse. Chunk #2 is rank-1
	// in sparse only. RRF: #1 = 2/61 ≈ 0.0328; #2 = 1/61 ≈ 0.0164. #1 wins.
	dense := []store.SearchResult{makeResult(1, "d")}
	sparse := []store.SearchResult{makeResult(1, "d"), makeResult(2, "d")}
	got := FuseRRF(dense, sparse, 10, 60)
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	if got[0].Chunk.ID != 1 {
		t.Errorf("consensus chunk #1 should rank first, got %d", got[0].Chunk.ID)
	}
	expected := 2.0 / 61.0
	if math.Abs(float64(got[0].Score)-expected) > 1e-6 {
		t.Errorf("consensus score = %f, want %f", got[0].Score, expected)
	}
}

func TestFuseRRF_RankPositionMatters(t *testing.T) {
	// Chunk #1: rank 1 in dense, rank 5 in sparse → 1/61 + 1/65 ≈ 0.0318
	// Chunk #2: rank 1 in sparse only → 1/61 ≈ 0.0164
	// #1 should still win because its dense rank-1 dominates.
	dense := []store.SearchResult{makeResult(1, "d")}
	sparse := []store.SearchResult{
		makeResult(2, "d"),
		makeResult(3, "d"),
		makeResult(4, "d"),
		makeResult(5, "d"),
		makeResult(1, "d"),
	}
	got := FuseRRF(dense, sparse, 10, 60)
	if got[0].Chunk.ID != 1 {
		t.Errorf("chunk #1 should win (consensus at rank 1 + 5), got top=%d", got[0].Chunk.ID)
	}
}

func TestFuseRRF_TopKClamp(t *testing.T) {
	dense := []store.SearchResult{
		makeResult(1, "d"), makeResult(2, "d"), makeResult(3, "d"),
		makeResult(4, "d"), makeResult(5, "d"),
	}
	got := FuseRRF(dense, nil, 2, 60)
	if len(got) != 2 {
		t.Errorf("topK=2 should clamp to 2 results, got %d", len(got))
	}
	if got[0].Chunk.ID != 1 || got[1].Chunk.ID != 2 {
		t.Errorf("top-2 should preserve rank order [1,2], got %v", ids(got))
	}
}

func TestFuseRRF_ZeroOrNegativeTopKReturnsEmpty(t *testing.T) {
	dense := []store.SearchResult{makeResult(1, "d")}
	for _, k := range []int{0, -1, -100} {
		got := FuseRRF(dense, nil, k, 60)
		if len(got) != 0 {
			t.Errorf("topK=%d should return empty, got %d results", k, len(got))
		}
	}
}

func TestFuseRRF_ScoreIsDescending(t *testing.T) {
	dense := []store.SearchResult{
		makeResult(1, "d"), makeResult(2, "d"), makeResult(3, "d"),
	}
	sparse := []store.SearchResult{
		makeResult(3, "d"), makeResult(4, "d"), makeResult(5, "d"),
	}
	got := FuseRRF(dense, sparse, 10, 60)
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("not descending at i=%d: %f > %f", i, got[i].Score, got[i-1].Score)
		}
	}
}

func TestFuseRRF_PreservesChunkAndDocumentMetadata(t *testing.T) {
	dense := []store.SearchResult{{
		Chunk:    domain.Chunk{ID: 7, DocID: "doc-A", Content: "alpha", SectionTitle: "Intro"},
		Document: domain.Document{ID: "doc-A", Title: "A title"},
	}}
	got := FuseRRF(dense, nil, 5, 60)
	if len(got) != 1 {
		t.Fatalf("expected 1 result, got %d", len(got))
	}
	if got[0].Chunk.Content != "alpha" || got[0].Chunk.SectionTitle != "Intro" {
		t.Errorf("chunk metadata lost: %+v", got[0].Chunk)
	}
	if got[0].Document.Title != "A title" {
		t.Errorf("document metadata lost: %+v", got[0].Document)
	}
}

// equalIDs checks slice equality for an int64 slice.
func equalIDs(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
