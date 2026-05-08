package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
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

	// Snapshot returns the current source registry, used to resolve a `tags`
	// argument to a set of source keys. Closure-based to keep the agent
	// package decoupled from the registry concretes (same pattern as
	// ListSourcesTool). When nil, the `tags` parameter cannot be resolved
	// and a non-empty `tags` arg falls into the empty-filter error path.
	Snapshot func() []domain.CrawlSource
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

// SetSnapshot wires in the source-registry accessor used to resolve `tags`
// to source keys. Pass nil to disable tag-based scoping (callers can still
// pass explicit source_keys).
func (t *SemanticSearchTool) SetSnapshot(fn func() []domain.CrawlSource) { t.Snapshot = fn }

type semanticSearchArgs struct {
	Query      string   `json:"query"`
	TopK       *int     `json:"top_k,omitempty"`
	SourceKeys []string `json:"source_keys,omitempty"`
	Tags       []string `json:"tags,omitempty"`
}

func (*SemanticSearchTool) Name() string { return "semantic_search" }

func (*SemanticSearchTool) Description() string {
	return "Semantic search over indexed documents. Use this to find information about ETSI courses, degrees, procedures, or other university information. For narrow topical questions, optionally scope retrieval by passing `source_keys` (specific sources) and/or `tags` (topic groups). Both come from list_sources; when both are provided their effects union (the search covers any source that matches either filter)."
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
				"description": "Optional. Restrict the search to documents reachable from any of these crawl-source keys (use list_sources to discover them). Combined with `tags` via union.",
			},
			"tags": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Optional. Restrict the search to documents whose source carries any of these tags (use list_sources to discover available tags). Multiple tags are ORed. Combined with `source_keys` via union.",
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

	scoped, ok := t.resolveScope(args.SourceKeys, args.Tags)
	if !ok {
		// User asked for a scope filter but it resolved to no sources. Don't
		// fall back to a full-corpus search — surface the mismatch so the
		// agent can correct the tags or drop the filter.
		return ToolResult{Text: emptyScopeMessage(args.SourceKeys, args.Tags)}, nil
	}

	results, err := t.Pipeline.Retrieve(ctx, args.Query, rag.RetrieveOpts{
		TopK:       args.TopK,
		SourceKeys: scoped,
	})
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

// resolveScope merges explicit source keys with the keys of every source
// whose Tags slice matches at least one of the requested tags (OR within
// `tags`, UNION between `tags` and `sourceKeys` — see the tool description
// for the rationale; this differs from the admin Documents page, which
// intersects).
//
// The boolean return flags whether the call should proceed. It is `false`
// only when the caller asked for a scope filter (one or both lists
// non-empty) but the union resolved to no sources — the caller should
// short-circuit instead of silently widening to a full-corpus search.
//
// (nil, true) means "no filter": pass through to the unscoped pipeline.
func (t *SemanticSearchTool) resolveScope(sourceKeys, tags []string) ([]string, bool) {
	if len(sourceKeys) == 0 && len(tags) == 0 {
		return nil, true
	}

	merged := make(map[string]struct{}, len(sourceKeys))
	for _, k := range sourceKeys {
		if k == "" {
			continue
		}
		merged[k] = struct{}{}
	}

	if len(tags) > 0 && t.Snapshot != nil {
		for _, k := range domain.SourceKeysWithAnyTag(t.Snapshot(), tags...) {
			merged[k] = struct{}{}
		}
	}

	if len(merged) == 0 {
		return nil, false
	}
	out := make([]string, 0, len(merged))
	for k := range merged {
		out = append(out, k)
	}
	return out, true
}

// emptyScopeMessage produces the agent-facing error text when a scope
// filter resolved to zero sources. The text names the offending args so
// the model can correct itself on the next turn.
func emptyScopeMessage(sourceKeys, tags []string) string {
	switch {
	case len(tags) > 0 && len(sourceKeys) > 0:
		return fmt.Sprintf("No sources match tags=%v and source_keys=%v. Call list_sources to see available keys and tags, or omit these parameters to search the full corpus.", tags, sourceKeys)
	case len(tags) > 0:
		return fmt.Sprintf("No sources match tags=%v. Call list_sources to see available tags, or omit `tags` to search the full corpus.", tags)
	default:
		return fmt.Sprintf("No sources match source_keys=%v. Call list_sources to see available keys, or omit `source_keys` to search the full corpus.", sourceKeys)
	}
}
