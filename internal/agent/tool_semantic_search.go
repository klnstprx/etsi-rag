package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"

	"etsi-rag/internal/markers"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
)

// SemanticSearchTool performs semantic search over indexed documents and
// returns both an LLM-facing context string and a set of citations for the UI.
// When configured with a Store, it also resolves [[image:ref]] markers in
// retrieved chunks to attached image artifacts (Images field on ToolResult).
type SemanticSearchTool struct {
	Pipeline *rag.Pipeline

	// Store provides artifact lookup for image markers; optional. Without it
	// the tool returns text-only.
	Store *store.SQLiteStore

	// MaxImages caps the number of image_url parts the agent attaches per
	// tool call. <= 0 disables image attachment.
	MaxImages int
}

// NewSemanticSearchTool constructs a semantic_search tool backed by the given
// RAG pipeline.
func NewSemanticSearchTool(pipeline *rag.Pipeline) *SemanticSearchTool {
	return &SemanticSearchTool{Pipeline: pipeline}
}

// SetStore enables image-marker resolution. Pass nil to disable.
func (t *SemanticSearchTool) SetStore(s *store.SQLiteStore) { t.Store = s }

// SetMaxImages caps how many image artifacts a single tool call attaches.
func (t *SemanticSearchTool) SetMaxImages(n int) { t.MaxImages = n }

type semanticSearchArgs struct {
	Query      string   `json:"query"`
	TopK       *int     `json:"top_k,omitempty"`
	SourceKeys []string `json:"source_keys,omitempty"`
}

func (*SemanticSearchTool) Name() string { return "semantic_search" }

func (*SemanticSearchTool) Description() string {
	return "Semantic search over indexed documents. Use this to find information about ETSI courses, degrees, procedures, or other university information. For narrow topical questions, optionally pass source_keys (from list_sources) to scope retrieval to a relevant subset of the corpus."
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
			"source_keys": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Optional. Restrict the search to documents reachable from any of these crawl-source keys (use list_sources to discover them). Omit to search the full corpus.",
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

	results, err := t.Pipeline.Retrieve(ctx, args.Query, args.TopK, args.SourceKeys...)
	if err != nil {
		return ToolResult{}, fmt.Errorf("RAG search: %w", err)
	}

	if len(results.Chunks) == 0 {
		return ToolResult{Text: "No relevant documents found."}, nil
	}

	// BuildContext is token-aware and always emits at least the top-1 chunk
	// (truncated if needed). Its output is our authoritative result — do not
	// fall back to an unbounded formatter.
	ragContext, err := t.Pipeline.BuildContext(ctx, results)
	if err != nil {
		return ToolResult{}, fmt.Errorf("building context: %w", err)
	}

	refs := rag.SourcesFromRetrieval(results)
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

	// Resolve image markers from the retrieved chunks to artifact URLs. The
	// LLM-visible text gets markers stripped (clean prose for the model);
	// the URLs are attached to ToolResult.Images for the agent loop to inject
	// as a synthetic user message when the backend supports vision.
	images := t.collectImages(ctx, results)
	llmText := markers.Strip(ragContext)

	return ToolResult{Text: llmText, Citations: citations, Images: images}, nil
}

// collectImages walks retrieved chunks for [[image:ref]] markers, looks up
// the matching artifact in the store, and builds path-only URLs that the
// agent can absolutize at injection time. Duplicates (same doc_id+ref) are
// deduped; total count is capped by MaxImages.
func (t *SemanticSearchTool) collectImages(ctx context.Context, results *rag.RetrievalResult) []ImageRef {
	if t.Store == nil || t.MaxImages <= 0 {
		return nil
	}

	type key struct{ doc, ref string }
	picked := make(map[key]struct{})
	var out []ImageRef

	for _, sc := range results.Chunks {
		for _, ref := range markers.Extract(sc.Chunk.Content) {
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
				// Marker without artifact: pdftoppm probably failed for this
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
