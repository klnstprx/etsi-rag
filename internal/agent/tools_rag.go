package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"etsi-rag/internal/domain"
	"etsi-rag/internal/rag"
	"etsi-rag/internal/store"

	"github.com/openai/openai-go"
)

// RAGSearchArgs holds the arguments for the rag_search tool.
type RAGSearchArgs struct {
	Query    string  `json:"query"`
	Language *string `json:"language,omitempty"`
	TopK     *int    `json:"top_k,omitempty"`
}

// NewRAGSearchTool creates a tool that searches the RAG knowledge base.
func NewRAGSearchTool(pipeline *rag.Pipeline) Tool {
	return Tool{
		Name:        "rag_search",
		Description: "Search the RAG knowledge base for relevant documents. Use this when you need to find information about ETSI courses, degrees, procedures, or other university information.",
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
		Execute: func(ctx context.Context, argsJSON string) (string, error) {
			var args RAGSearchArgs
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

			return formatRAGResults(results), nil
		},
	}
}

// ListDocumentsArgs holds the arguments for the list_documents tool.
type ListDocumentsArgs struct {
	Language *string `json:"language,omitempty"`
}

// NewListDocumentsTool creates a tool that lists indexed documents.
func NewListDocumentsTool(s *store.SQLiteStore) Tool {
	return Tool{
		Name:        "list_documents",
		Description: "List all indexed documents in the knowledge base.",
		Parameters: openai.FunctionParameters{
			"type": "object",
			"properties": map[string]any{
				"language": map[string]any{
					"type":        "string",
					"enum":        []string{"es", "en"},
					"description": "Filter by language (optional)",
				},
			},
		},
		Execute: func(ctx context.Context, argsJSON string) (string, error) {
			var args ListDocumentsArgs
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return "", fmt.Errorf("parsing args: %w", err)
			}

			var lang *domain.Language
			if args.Language != nil {
				l := domain.Language(*args.Language)
				lang = &l
			}

			docs, err := s.ListDocuments(ctx, lang)
			if err != nil {
				return "", fmt.Errorf("listing documents: %w", err)
			}

			if len(docs) == 0 {
				return "No documents found.", nil
			}

			var b strings.Builder
			for i, doc := range docs {
				fmt.Fprintf(&b, "%d. %s (ID: %s, Language: %s, Category: %s)\n",
					i+1, doc.Title, doc.ID, doc.Language, doc.Category)
			}
			return b.String(), nil
		},
	}
}

func formatRAGResults(results *rag.RetrievalResult) string {
	var b strings.Builder
	docTitles := make(map[string]string)
	for _, d := range results.Documents {
		docTitles[d.ID] = d.Title
	}

	for i, sc := range results.Chunks {
		title := docTitles[sc.Chunk.DocID]
		fmt.Fprintf(&b, "[%d] %s (score: %.2f):\n%s\n\n", i+1, title, sc.Score, sc.Chunk.Content)
	}
	return b.String()
}
