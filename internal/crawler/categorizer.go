package crawler

import (
	"strings"

	"etsi-rag/internal/domain"
)

// Categorize determines the document category from a URL.
// Falls back to the source's default category, then to CategoryGeneral.
func Categorize(url, sourceCategory string) domain.Category {
	lower := strings.ToLower(url)

	switch {
	case strings.Contains(lower, "/centers/subject/") || strings.Contains(lower, "programasasignaturas"):
		return domain.CategoryCourse
	case strings.Contains(lower, "programacion-docente"):
		return domain.CategoryCourse
	case strings.Contains(lower, "tramit") || strings.Contains(lower, "matricula") || strings.Contains(lower, "admision"):
		return domain.CategoryProcedure
	case strings.Contains(lower, "personal") || strings.Contains(lower, "staff"):
		return domain.CategoryStaff
	}

	if cat := domain.Category(sourceCategory); cat.Valid() {
		return cat
	}
	return domain.CategoryGeneral
}

// StableDocID generates a content-addressed document ID from the SHA-256
// content hash hex string. Two URLs with identical extracted text map to the
// same ID, so duplicate content is stored once and shared across crawl_pages.
func StableDocID(contentHash string) string {
	return "crawl-" + contentHash[:24]
}
