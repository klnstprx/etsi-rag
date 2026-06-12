package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
)

// SemanticSearchTool performs semantic search over indexed documents and
// returns both an LLM-facing context string and a set of citations for the UI.
// When configured with a Store, it also resolves [[image:ref]] markers in
// retrieved chunks to attached image artifacts (Images field on ToolResult).
type SemanticSearchTool struct {
	Retriever *rag.Retriever

	// Store provides artifact lookup for image markers; optional. Without it
	// the tool returns text-only.
	Store *store.SQLiteStore

	// MaxImages caps the number of image_url parts the agent attaches per
	// tool call. <= 0 disables image attachment.
	MaxImages int
}

// NewSemanticSearchTool constructs a semantic_search tool backed by the given
// retriever.
func NewSemanticSearchTool(r *rag.Retriever) *SemanticSearchTool {
	return &SemanticSearchTool{Retriever: r}
}

// SetStore enables image-marker resolution. Pass nil to disable.
func (t *SemanticSearchTool) SetStore(s *store.SQLiteStore) { t.Store = s }

// SetMaxImages caps how many image artifacts a single tool call attaches.
func (t *SemanticSearchTool) SetMaxImages(n int) { t.MaxImages = n }

type semanticSearchArgs struct {
	Query string `json:"query"`
	TopK  *int   `json:"top_k,omitempty"`
}

func (*SemanticSearchTool) Name() string { return "semantic_search" }

func (*SemanticSearchTool) Description() string {
	return "Semantic search over indexed documents. Use this to find information about ETSI courses, degrees, procedures, or other university information."
}

func (*SemanticSearchTool) Parameters() openai.FunctionParameters {
	return openai.FunctionParameters{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "The search query",
			},
			"top_k": map[string]any{
				"type":        "integer",
				"description": "Number of results to return (optional; server default is used when omitted).",
			},
		},
		"required": []string{"query"},
	}
}

func (t *SemanticSearchTool) Execute(ctx context.Context, argsJSON string) (ToolResult, error) {
	var args semanticSearchArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return ToolResult{}, fmt.Errorf("parsing args: %w", err)
	}

	results, err := t.Retriever.Retrieve(ctx, args.Query, rag.RetrieveOpts{
		TopK: args.TopK,
	})
	if err != nil {
		return ToolResult{}, fmt.Errorf("RAG search: %w", err)
	}

	if len(results.Chunks) == 0 {
		return ToolResult{Text: "No relevant documents found."}, nil
	}

	// BuildContext is token-aware: it packs whole chunks in rank order until
	// the budget is hit, and returns an empty context if even the top-1 chunk
	// does not fit. Its output is our authoritative result — do not fall back
	// to an unbounded formatter.
	ragContext, err := t.Retriever.BuildContext(ctx, results)
	if err != nil {
		return ToolResult{}, fmt.Errorf("building context: %w", err)
	}

	refs := rag.SourcesFromRetrieval(results)
	citations := dedupCitations(refs)

	// Chunk content is already marker-free (the chunker strips markers during
	// emission); ImageRefs lives on the chunk record. Resolve refs to artifact
	// URLs so the agent loop can inject them as a synthetic user message when
	// the backend supports vision.
	images := t.collectImages(ctx, results)

	return ToolResult{Text: ragContext, Citations: citations, Images: images}, nil
}

// dedupCitations converts source references into UI-facing citations,
// deduplicating by URL when present and falling back to the reference ID for
// entries without a URL. Order of first occurrence is preserved.
func dedupCitations(refs []domain.SourceReference) []Citation {
	citations := make([]Citation, 0, len(refs))
	seen := make(map[string]struct{}, len(refs))
	for _, r := range refs {
		key := r.ID
		if r.URL != nil && *r.URL != "" {
			key = *r.URL
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}

		c := Citation{Title: r.Title}
		if r.URL != nil {
			c.URL = *r.URL
		}
		citations = append(citations, c)
	}
	return citations
}

// collectImages walks retrieved chunks' ImageRefs (stamped by the chunker at
// ingest time), looks up each (doc_id, ref) pair in the store, and builds
// path-only URLs that the agent can absolutize at injection time. Duplicates
// (same doc_id+ref) are deduped; total count is capped by MaxImages.
func (t *SemanticSearchTool) collectImages(ctx context.Context, results *rag.RetrievalResult) []ImageRef {
	if t.Store == nil || t.MaxImages <= 0 {
		return nil
	}

	type key struct{ doc, ref string }
	picked := make(map[key]struct{})
	var out []ImageRef

	for _, sc := range results.Chunks {
		for _, ref := range sc.Chunk.ImageRefs {
			k := key{doc: sc.Chunk.DocID, ref: ref}
			if _, dup := picked[k]; dup {
				continue
			}
			picked[k] = struct{}{}

			art, err := t.Store.GetArtifact(ctx, sc.Chunk.DocID, ref)
			if err != nil {
				if !errors.Is(err, store.ErrNotFound) {
					// Storage error is not fatal — log via the result text isn't
					// useful, the agent layer just sees fewer images.
					continue
				}
				// Ref without artifact: pdftoppm probably failed for this
				// page, or backfill hasn't run yet. Skip silently.
				continue
			}

			out = append(out, ImageRef{
				DocID: sc.Chunk.DocID,
				Ref:   ref,
				URL:   "/v1/artifacts/" + art.PublicToken + ".png",
			})
			if len(out) >= t.MaxImages {
				return out
			}
		}
	}
	return out
}
