package rag

import (
	"sort"

	"etsi-rag/internal/store"
)

// rrfK is the canonical RRF constant from Cormack et al. (see FuseRRF).
const rrfK = 60.0

// FuseRRF merges two ranked SearchResult lists by Reciprocal Rank Fusion.
//
// For each input list, the rank-i contribution to a chunk's fused score is
// 1/(k+i) where i starts at 1. A chunk that appears in both lists receives
// both contributions; one that appears in only one gets a single
// contribution. RRF ignores absolute scores — it uses positional rank only —
// so it works equally well for cosine (∈[0,1]) and BM25 (∈ℝ) without
// requiring score normalisation.
//
// Output is sorted by descending fused score and clamped to topK. The
// `Score` field on each returned result is the fused score (also "higher is
// better"). `k=60` is the canonical RRF constant from the original paper.
//
// Source:
//
//	Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal
//	Rank Fusion outperforms Condorcet and individual Rank Learning
//	Methods. SIGIR '09, pp. 758–759. doi:10.1145/1571941.1572114
//
// See docs/retrieval.md for formula, parameter rationale, and the wider
// hybrid-retrieval context.
func FuseRRF(dense, sparse []store.SearchResult, topK int, k float64) []store.SearchResult {
	if topK <= 0 {
		return nil
	}

	type fused struct {
		result store.SearchResult
		score  float64
	}

	merged := make(map[int64]*fused)
	contribute := func(list []store.SearchResult) {
		for i, r := range list {
			rank := i + 1
			contribution := 1.0 / (k + float64(rank))
			if existing, ok := merged[r.Chunk.ID]; ok {
				existing.score += contribution
				continue
			}
			merged[r.Chunk.ID] = &fused{result: r, score: contribution}
		}
	}
	contribute(dense)
	contribute(sparse)

	if len(merged) == 0 {
		return nil
	}

	results := make([]store.SearchResult, 0, len(merged))
	for _, f := range merged {
		out := f.result
		out.Score = float32(f.score)
		results = append(results, out)
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if topK < len(results) {
		results = results[:topK]
	}
	return results
}
