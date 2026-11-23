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

// subjectKey returns "{plan}/{code}" for a /centers/subject/ URL, so the
// variants of one plan's subject share a key while the same course under
// another plan, which links its own guía PDF, does not. Returns "" for
// non-subject URLs.
//
// URL patterns:
//   - /centers/subject/{plan}/{code}                 → plan/code
//   - /centers/subject/{plan}/{code}/                → plan/code
//   - /centers/subject/{center}/{plan}/{code}/{year} → plan/code
func subjectKey(rawURL string) string {
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

	// The plan and code are the first two numeric segments; a center is a slug.
	var numerics []string
	for _, p := range parts {
		if isNumeric(p) {
			numerics = append(numerics, p)
		}
	}

	if len(numerics) < 2 {
		return ""
	}
	return numerics[0] + "/" + numerics[1]
}

// isNumeric returns true if s is a non-empty string of ASCII digits.
func isNumeric(s string) bool {
	return s != "" && !strings.ContainsFunc(s, func(r rune) bool { return r < '0' || r > '9' })
}

// shouldFollow decides whether a discovered link should be crawled.
func shouldFollow(link, seedPath string) bool {
	parsed, err := url.Parse(link)
	if err != nil {
		return false
	}

	host := strings.ToLower(parsed.Hostname())
	path := strings.ToLower(parsed.Path)

	if hasBlockedExtension(path) {
		return false
	}

	switch host {
	case "www.uma.es":
		// Skip news under every prefix, so it never depends on which source finds it.
		if isNewsPath(path) {
			return false
		}
		// Follow: seed path prefix, etsi-informatica, subject pages, media files
		return strings.HasPrefix(path, seedPath) ||
			strings.HasPrefix(path, "/etsi-informatica/") ||
			strings.HasPrefix(path, "/centers/subject/") ||
			strings.HasPrefix(path, "/media/files/") ||
			strings.HasPrefix(path, "/media/tinyimages/file/")

	case "sara.uma.es":
		// Only follow guía docente PDFs
		return strings.Contains(path, "programasasignaturas")

	default:
		return false
	}
}

// isNewsPath reports whether a lowercased path has a news segment, as in
// /etsi-informatica/noticias/{slug}/ or the file browser's Noticias folder. A
// file name that only mentions news, such as a degree plan's "Anexo I.
// Noticias relevantes.pdf", is not news.
func isNewsPath(path string) bool {
	for _, seg := range strings.Split(path, "/") {
		switch seg {
		case "noticias", "news", "newsitem":
			return true
		}
	}
	return false
}

// blockedExtensions contains file extensions that should never be crawled.
// These are binary or non-text formats that can't be meaningfully embedded.
var blockedExtensions = map[string]bool{
	// Office documents (non-PDF)
	".docx": true, ".dotx": true, ".doc": true,
	".pptx": true, ".ppt": true,
	".xlsx": true, ".xls": true,
	// Images
	".jpg": true, ".jpeg": true, ".png": true, ".gif": true,
	".svg": true, ".webp": true, ".ico": true, ".bmp": true, ".tiff": true,
	// Archives
	".zip": true, ".rar": true, ".gz": true, ".tar": true, ".7z": true,
	// Media
	".mp3": true, ".mp4": true, ".avi": true, ".mov": true, ".wmv": true,
	".wav": true, ".flac": true, ".ogg": true,
	// Executables
	".exe": true, ".msi": true, ".dmg": true, ".apk": true,
}

// hasBlockedExtension returns true if the URL path ends with a blocked file extension.
func hasBlockedExtension(path string) bool {
	lower := strings.ToLower(path)
	// Find last dot in the path (ignore query string)
	if idx := strings.LastIndex(lower, "."); idx >= 0 {
		ext := lower[idx:]
		return blockedExtensions[ext]
	}
	return false
}
