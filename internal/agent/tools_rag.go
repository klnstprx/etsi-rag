package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/rag"

	"github.com/openai/openai-go"
)

// SemanticSearchArgs holds the arguments for the semantic_search tool.
type SemanticSearchArgs struct {
	Query    string  `json:"query"`
	Language *string `json:"language,omitempty"`
	TopK     *int    `json:"top_k,omitempty"`
}

// NewSemanticSearchTool creates a tool that performs semantic search over indexed documents.
// It uses BuildContext to format results and collects source references.
func NewSemanticSearchTool(pipeline *rag.Pipeline) Tool {
	return Tool{
		Name:        "semantic_search",
		Description: "Semantic search over indexed documents. Use this to find information about ETSI courses, degrees, procedures, or other university information.",
		Parameters: openai.FunctionParameters{
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
		},
		Execute: func(ctx context.Context, argsJSON string, sources *[]domain.SourceReference) (string, error) {
			var args SemanticSearchArgs
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return "", fmt.Errorf("parsing args: %w", err)
			}

			results, err := pipeline.Retrieve(ctx, args.Query, args.Language, args.TopK)
			if err != nil {
				return "", fmt.Errorf("RAG search: %w", err)
			}

			if len(results.Chunks) == 0 {
				return "No relevant documents found.", nil
			}

			// Collect sources
			if sources != nil {
				*sources = append(*sources, rag.SourcesFromRetrieval(results)...)
			}

			// BuildContext is token-aware and always emits at least the top-1
			// chunk (truncated if needed). Its output is our authoritative
			// result — do not fall back to an unbounded formatter.
			ragContext, err := pipeline.BuildContext(ctx, results)
			if err != nil {
				return "", fmt.Errorf("building context: %w", err)
			}
			return ragContext, nil
		},
	}
}
