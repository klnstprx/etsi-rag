package agent

import (
	"context"
	"encoding/json"
	"testing"

	"etsi-rag/internal/domain"
)

func TestListSourcesTool_Execute(t *testing.T) {
	sources := []domain.CrawlSource{
		{
			Key:         "becas",
			Name:        "Becas y ayudas UMA",
			SeedURL:     "https://www.uma.es/becas/",
			Description: "Convocatorias MEFPYD, becas UMA, becas de movilidad.",
			Tags:        []string{"financial-aid", "mobility"},
			Enabled:     true,
		},
		{
			Key:     "disabled",
			Name:    "Disabled Source",
			SeedURL: "https://www.uma.es/disabled/",
			Tags:    []string{"x"},
			Enabled: false,
		},
		{
			Key:     "no-tags",
			Name:    "No Tags Source",
			SeedURL: "https://www.uma.es/notags/",
			Enabled: true,
		},
	}
	tool := NewListSourcesTool(func() []domain.CrawlSource { return sources })

	result, err := tool.Execute(context.Background(), "{}")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(result.Citations) != 0 {
		t.Errorf("expected no citations, got %d", len(result.Citations))
	}
	if len(result.Images) != 0 {
		t.Errorf("expected no images, got %d", len(result.Images))
	}

	var entries []listSourcesEntry
	if err := json.Unmarshal([]byte(result.Text), &entries); err != nil {
		t.Fatalf("unmarshal text: %v\ntext=%s", err, result.Text)
	}
	if len(entries) != 2 {
		t.Fatalf("want 2 enabled sources, got %d (%v)", len(entries), entries)
	}

	if entries[0].Key != "becas" || entries[0].Description == "" {
		t.Errorf("entry[0] = %+v", entries[0])
	}
	if len(entries[0].Tags) != 2 || entries[0].Tags[0] != "financial-aid" {
		t.Errorf("entry[0].Tags = %v", entries[0].Tags)
	}
	if entries[1].Key != "no-tags" {
		t.Errorf("entry[1] = %+v", entries[1])
	}
	if len(entries[1].Tags) != 0 {
		t.Errorf("entry[1] should omit tags, got %v", entries[1].Tags)
	}
}

func TestListSourcesTool_NilSnapshot(t *testing.T) {
	tool := NewListSourcesTool(nil)
	result, err := tool.Execute(context.Background(), "{}")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Text != "[]" {
		t.Errorf("nil snapshot should return empty array, got %q", result.Text)
	}
}

func TestListSourcesTool_StaticMetadata(t *testing.T) {
	tool := NewListSourcesTool(func() []domain.CrawlSource { return nil })
	if tool.Name() != "list_sources" {
		t.Errorf("Name = %q, want list_sources", tool.Name())
	}
	if tool.Description() == "" {
		t.Error("Description must not be empty")
	}
	params := tool.Parameters()
	if params["type"] != "object" {
		t.Errorf("Parameters.type = %v, want object", params["type"])
	}
}
