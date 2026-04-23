package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"

	"etsi-rag/internal/rag"
)

// SemanticSearchTool performs semantic search over indexed documents and
// returns both an LLM-facing context string and a set of citations for the UI.
type SemanticSearchTool struct {
	Pipeline *rag.Pipeline
}

// NewSemanticSearchTool constructs a semantic_search tool backed by the given
// RAG pipeline.
func NewSemanticSearchTool(pipeline *rag.Pipeline) *SemanticSearchTool {
	return &SemanticSearchTool{Pipeline: pipeline}
}

type semanticSearchArgs struct {
	Query    string  `json:"query"`
	Language *string `json:"language,omitempty"`
	TopK     *int    `json:"top_k,omitempty"`
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
			"language": map[string]any{
				"type":        "string",
				"enum":        []string{"es", "en"},
				"description": "Language filter (optional)",
			},
			"top_k": map[string]any{
				"type":        "integer",
				"description": "Number of results to return (default: 5)",
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

	results, err := t.Pipeline.Retrieve(ctx, args.Query, args.Language, args.TopK)
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

	return ToolResult{Text: ragContext, Citations: citations}, nil
}
