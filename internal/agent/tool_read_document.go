package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/markers"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"
)

// ReadDocumentTool fetches the full content of an indexed document by URL,
// or — when a query is provided — runs a doc-scoped dense search that
// returns only the most relevant chunks within that document.
//
// This is the agent's escape hatch when a previous semantic_search snippet
// was too narrow to answer the question. The roadmap originally framed this
// as "RustFS-anchored", but the clean markdown is already in
// documents.content (and the structure-aware chunker exposes per-section
// metadata via chunks.section_title), so re-extracting raw HTML/PDF on each
// call would be needless work and would lose section structure. RustFS keeps
// earning its place via multimodal artifacts; for text expansion the chunks
// table is the right source.
//
// The token-budget guard in agent.checkToolResult is the first line of
// defence against oversized whole-document replies; the LLM's natural retry
// is to add a query parameter to scope the result.
type ReadDocumentTool struct {
	Pipeline *rag.Pipeline
	Store    *store.SQLiteStore

	// DefaultTopK is the top_k used in query mode when the LLM doesn't
	// supply one. Zero (or unset) falls back to 5, mirroring the typical
	// retrieval.top_k.
	DefaultTopK int
}

// NewReadDocumentTool constructs a read_document tool. Both the pipeline
// (for query-mode embedding + scoped search) and the store (for whole-doc
// fetch and URL → doc resolution) are required.
func NewReadDocumentTool(p *rag.Pipeline, s *store.SQLiteStore) *ReadDocumentTool {
	return &ReadDocumentTool{Pipeline: p, Store: s}
}

// SetDefaultTopK overrides the default top_k used when the LLM doesn't pass
// one in query mode. Wired from the retrieval config at startup.
func (t *ReadDocumentTool) SetDefaultTopK(k int) { t.DefaultTopK = k }

type readDocumentArgs struct {
	URL   string `json:"url"`
	Query string `json:"query,omitempty"`
	TopK  *int   `json:"top_k,omitempty"`
}

func (*ReadDocumentTool) Name() string { return "read_document" }

func (*ReadDocumentTool) Description() string {
	return "Read the full content of an indexed document by URL. Use this when a previous semantic_search snippet did not contain enough context to answer. Pass a `url` from a citation. If the document is too long for the context budget, also pass a `query` to scope the result to the most relevant sections within that document."
}

func (*ReadDocumentTool) Parameters() openai.FunctionParameters {
	return openai.FunctionParameters{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{
				"type":        "string",
				"description": "Document URL exactly as it appeared in a previous citation.",
			},
			"query": map[string]any{
				"type":        "string",
				"description": "Optional. When provided, returns only the top-K chunks within the document that match this query (semantic search scoped to one document).",
			},
			"top_k": map[string]any{
				"type":        "integer",
				"description": "Optional. Number of chunks to return when `query` is set. Default uses the server's retrieval top_k.",
			},
		},
		"required": []string{"url"},
	}
}

func (t *ReadDocumentTool) Execute(ctx context.Context, argsJSON string) (ToolResult, error) {
	var args readDocumentArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return ToolResult{}, fmt.Errorf("parsing args: %w", err)
	}
	if args.URL == "" {
		return ToolResult{Text: "Missing required parameter `url`."}, nil
	}

	doc, err := t.Store.GetDocumentByURL(ctx, args.URL)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return ToolResult{
				Text: fmt.Sprintf("Document with URL %q not found in the index.", args.URL),
			}, nil
		}
		return ToolResult{}, fmt.Errorf("looking up document: %w", err)
	}

	if strings.TrimSpace(args.Query) == "" {
		return t.wholeDocument(doc), nil
	}
	return t.scopedQuery(ctx, doc, args.Query, args.TopK)
}

// wholeDocument returns the full markdown body of the document. Markers are
// stripped from the LLM-visible text — image attachment for whole-doc mode
// is out of scope for the first cut (semantic_search already handles vision
// when the model needs it).
func (t *ReadDocumentTool) wholeDocument(doc *domain.Document) ToolResult {
	cleaned := markers.Strip(doc.Content)
	if cleaned == "" {
		cleaned = "(document is empty)"
	}
	return ToolResult{
		Text:      cleaned,
		Citations: []Citation{citationFor(doc)},
	}
}

// scopedQuery runs a dense search within the document and formats the top-K
// chunks the same way Pipeline.BuildContext does, so the LLM sees a
// consistent shape across both tools.
func (t *ReadDocumentTool) scopedQuery(ctx context.Context, doc *domain.Document, query string, topK *int) (ToolResult, error) {
	k := t.DefaultTopK
	if k <= 0 {
		k = 5
	}
	if topK != nil && *topK > 0 {
		k = *topK
	}

	results, err := t.Pipeline.RetrieveInDocument(ctx, query, doc.ID, k)
	if err != nil {
		return ToolResult{}, fmt.Errorf("scoped search: %w", err)
	}
	if results == nil || len(results.Chunks) == 0 {
		return ToolResult{
			Text:      fmt.Sprintf("Document %q found, but no sections matched the query %q.", doc.Title, query),
			Citations: []Citation{citationFor(doc)},
		}, nil
	}

	var b strings.Builder
	for i, sc := range results.Chunks {
		header := doc.Title
		if sc.Chunk.SectionTitle != "" {
			header = doc.Title + " > " + sc.Chunk.SectionTitle
		}
		fmt.Fprintf(&b, "[%d] %s:\n%s\n\n", i+1, header, sc.Chunk.Content)
	}
	return ToolResult{
		Text:      b.String(),
		Citations: []Citation{citationFor(doc)},
	}, nil
}

// citationFor builds the single Citation that read_document emits. The URL
// may be empty (NULL in DB), in which case we leave it empty rather than
// fabricating one.
func citationFor(doc *domain.Document) Citation {
	c := Citation{Title: doc.Title}
	if doc.URL != nil {
		c.URL = *doc.URL
	}
	return c
}
