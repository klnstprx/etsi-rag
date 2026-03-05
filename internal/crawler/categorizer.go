package crawler

import (
	"crypto/sha256"
	"fmt"
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

// StableDocID generates a deterministic document ID from a URL.
func StableDocID(url string) string {
	h := sha256.Sum256([]byte(url))
	return fmt.Sprintf("crawl-%x", h[:12]) // 24 hex chars
}
