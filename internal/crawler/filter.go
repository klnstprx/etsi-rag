package crawler

import (
	"net/url"
	"strings"
)

// isLowQualityContent returns true if the extracted text should be discarded.
// It checks three things:
//   - Content length below minContentLen (0 disables the length check)
//   - Configurable garbage patterns (auth errors, soft 404s, stubs)
//   - Empty filter pages (short pages starting with "Filtrar por curso:")
//
// Patterns are matched case-insensitively as substrings and always reject
// regardless of length. A nil/empty patterns slice disables the pattern check.
func isLowQualityContent(text string, minContentLen int, patterns []string) bool {
	if minContentLen > 0 && len(text) < minContentLen {
		return true
	}

	lower := strings.ToLower(text)
	for _, pat := range patterns {
		if strings.Contains(lower, strings.ToLower(pat)) {
			return true
		}
	}

	if strings.HasPrefix(text, "Filtrar por curso:") && len(text) < 200 {
		return true
	}

	return false
}

// subjectCode extracts the numeric subject code from a /centers/subject/ URL.
// Returns "" for non-subject URLs.
//
// URL patterns:
//   - /centers/subject/{plan}/{code}         → code
//   - /centers/subject/{plan}/{code}/        → code
//   - /centers/subject/{center}/{plan}/{code}/{year} → code (second numeric segment)
func subjectCode(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}

	path := strings.TrimSuffix(parsed.Path, "/")
	const prefix = "/centers/subject/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}

	rest := path[len(prefix):]
	parts := strings.Split(rest, "/")
	if len(parts) < 2 {
		return ""
	}

	// Find numeric segments — the subject code is always the second numeric value.
	// /centers/subject/{plan}/{code} → parts[0]=plan(numeric), parts[1]=code(numeric)
	// /centers/subject/{center}/{plan}/{code}/{year} → parts[1]=plan(numeric), parts[2]=code(numeric)
	var numerics []string
	for _, p := range parts {
		if isNumeric(p) {
			numerics = append(numerics, p)
		}
	}

	if len(numerics) < 2 {
		return ""
	}
	return numerics[1]
}

// isNumeric returns true if s is a non-empty string of digits.
func isNumeric(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
