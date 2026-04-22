package crawler

import (
	"testing"

	"etsi-rag/internal/domain"
)

func TestCategorize(t *testing.T) {
	tests := []struct {
		name           string
		url            string
		sourceCategory string
		want           domain.Category
	}{
		// URL pattern matches
		{"subject page", "https://www.uma.es/centers/subject/1234", "general", domain.CategoryCourse},
		{"guia docente PDF", "https://sara.uma.es/ProgramasAsignaturas_5102.pdf", "", domain.CategoryCourse},
		{"programacion docente", "https://www.uma.es/etsi-informatica/programacion-docente/", "general", domain.CategoryCourse},
		{"tramite page", "https://www.uma.es/etsi-informatica/tramites/matricula-info", "general", domain.CategoryProcedure},
		{"matricula page", "https://www.uma.es/etsi-informatica/matricula-grado/", "", domain.CategoryProcedure},
		{"admision page", "https://www.uma.es/etsi-informatica/admision/", "", domain.CategoryProcedure},
		{"staff page", "https://www.uma.es/etsi-informatica/personal/", "general", domain.CategoryStaff},
		{"staff english", "https://www.uma.es/etsi-informatica/staff/members", "", domain.CategoryStaff},
		// Fallback to source category
		{"fallback to source category", "https://www.uma.es/etsi-informatica/some-page", "course", domain.CategoryCourse},
		// Fallback to general
		{"fallback to general with invalid source", "https://www.uma.es/etsi-informatica/some-page", "invalid", domain.CategoryGeneral},
		{"fallback to general with empty source", "https://www.uma.es/etsi-informatica/some-page", "", domain.CategoryGeneral},
		// Case insensitivity
		{"case insensitive URL", "https://www.uma.es/Centers/Subject/1234", "general", domain.CategoryCourse},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Categorize(tt.url, tt.sourceCategory)
			if got != tt.want {
				t.Errorf("Categorize(%q, %q) = %q, want %q", tt.url, tt.sourceCategory, got, tt.want)
			}
		})
	}
}

func TestStableDocID(t *testing.T) {
	// Arbitrary but valid-shaped SHA-256 hex strings.
	hashA := "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	hashB := "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"

	t.Run("same content hash produces same ID", func(t *testing.T) {
		id1 := StableDocID(hashA)
		id2 := StableDocID(hashA)
		if id1 != id2 {
			t.Errorf("StableDocID not deterministic: %q != %q", id1, id2)
		}
	})

	t.Run("different content hashes produce different IDs", func(t *testing.T) {
		if StableDocID(hashA) == StableDocID(hashB) {
			t.Error("different content hashes produced same ID")
		}
	})

	t.Run("format is crawl- prefix plus 24 hex chars", func(t *testing.T) {
		id := StableDocID(hashA)
		if len(id) != len("crawl-")+24 {
			t.Errorf("unexpected ID length: got %d (%q)", len(id), id)
		}
		if id[:6] != "crawl-" {
			t.Errorf("missing crawl- prefix: %q", id)
		}
	})
}
