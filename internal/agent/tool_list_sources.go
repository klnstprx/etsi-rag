package agent

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go"

	"etsi-rag/internal/domain"
)

// ListSourcesTool exposes the operator-curated source catalogue to the agent.
// The agent reads the description and tags of each enabled source to decide
// whether to scope a follow-up semantic_search by passing source_keys, or to
// search the whole corpus.
//
// The catalogue is loaded fresh on every Execute via the injected snapshot
// function, so admin edits to sources.toml (which hot-reload through
// crawler.SetSources) take effect for the next agent turn without rebuilding
// the agent.
type ListSourcesTool struct {
	// Snapshot returns the current source list. Closure-based so the agent
	// package stays decoupled from crawler / config concretes.
	Snapshot func() []domain.CrawlSource
}

// NewListSourcesTool constructs a list_sources tool. The snapshot function
// must be safe to call concurrently and should return a fresh, defensively
// copied slice each call.
func NewListSourcesTool(snapshot func() []domain.CrawlSource) *ListSourcesTool {
	return &ListSourcesTool{Snapshot: snapshot}
}

// listSourcesEntry is the on-the-wire shape of one source as the LLM sees it.
// Tags is omitted when empty so the JSON the model reads stays compact.
type listSourcesEntry struct {
	Key         string   `json:"key"`
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

func (*ListSourcesTool) Name() string { return "list_sources" }

func (*ListSourcesTool) Description() string {
	return "List the operator-curated knowledge-base sources, each with a description and topic tags. Call this when the user asks a narrow topical question (e.g. about scholarships, mobility, exam dates, library services) and you want to scope a follow-up semantic_search to a relevant subset of the corpus by passing source_keys. For broad questions about a degree or general topics, you can skip this tool and search the whole corpus."
}

func (*ListSourcesTool) Parameters() openai.FunctionParameters {
	return openai.FunctionParameters{
		"type":       "object",
		"properties": map[string]any{},
	}
}

func (t *ListSourcesTool) Execute(ctx context.Context, _ string) (ToolResult, error) {
	if t.Snapshot == nil {
		return ToolResult{Text: "[]"}, nil
	}
	sources := t.Snapshot()

	entries := make([]listSourcesEntry, 0, len(sources))
	for _, s := range sources {
		if !s.Enabled {
			continue
		}
		entries = append(entries, listSourcesEntry{
			Key:         s.Key,
			Name:        s.Name,
			Description: s.Description,
			Tags:        s.Tags,
		})
	}

	body, err := json.Marshal(entries)
	if err != nil {
		// Marshal of plain strings/slices cannot realistically fail; fall back
		// to a minimal manual response so the agent isn't blocked.
		return ToolResult{Text: "[]"}, nil
	}
	return ToolResult{Text: string(body)}, nil
}
