package crawler

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"unicode"

	"etsi-rag/internal/config"
)

var pageNumberRe = regexp.MustCompile(`(?im)^\s*p[aá]gina\s+\d+\s+de\s*(\s+\d+)?\s*$`)

// spanishStopWords are lowercased in title case conversion (except when first word).
var spanishStopWords = map[string]bool{
	"de": true, "la": true, "el": true, "los": true, "las": true,
	"del": true, "en": true, "y": true, "o": true, "al": true,
	"por": true, "con": true, "para": true, "una": true, "un": true,
	"su": true, "sus": true, "e": true, "a": true,
}

// PDFResult contains the extracted content from a PDF file.
type PDFResult struct {
	Title string
	Text  string // markdown-formatted
}

// ExtractPDFText extracts text from a PDF using pdftotext (poppler) and
// converts it to structured markdown with heading detection.
func ExtractPDFText(ctx context.Context, pdfData []byte, cfg config.PDFConfig) (*PDFResult, error) {
	tmp, err := os.CreateTemp("", "crawl-*.pdf")
	if err != nil {
		return nil, fmt.Errorf("creating temp file: %w", err)
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(pdfData); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("writing temp file: %w", err)
	}
	tmp.Close()

	// Run pdftotext in default (reading order) mode — no -layout flag.
	// Default mode undoes column formatting and reads in logical order.
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "pdftotext", "-enc", "UTF-8", tmp.Name(), "-")
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("pdftotext: %w: %s", err, stderr.String())
	}

	title := extractPDFTitle(ctx, tmp.Name())
	text := pdfTextToMarkdown(stdout.String(), cfg)

	return &PDFResult{Title: title, Text: text}, nil
}

// extractPDFTitle runs pdfinfo and extracts the Title metadata field.
// Returns empty string on any error (non-fatal).
func extractPDFTitle(ctx context.Context, pdfPath string) string {
	var stdout bytes.Buffer
	cmd := exec.CommandContext(ctx, "pdfinfo", pdfPath)
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		return ""
	}

	for line := range strings.SplitSeq(stdout.String(), "\n") {
		title, _ := strings.CutPrefix(line, "Title:")
		title = strings.TrimSpace(title)
		if title != "" {
			return title
		}
	}
	return ""
}

// pdfTextToMarkdown converts raw pdftotext output to structured markdown.
// Pipeline: strip page headers → detect headings → normalize.
func pdfTextToMarkdown(raw string, cfg config.PDFConfig) string {
	if raw == "" {
		return ""
	}

	text := stripPageHeaders(raw)

	var sb strings.Builder
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if isHeadingLine(trimmed, cfg.HeadingPatterns) {
			sb.WriteString("## ")
			sb.WriteString(toTitleCase(trimmed))
			sb.WriteByte('\n')
		} else {
			sb.WriteString(line)
			sb.WriteByte('\n')
		}
	}

	return normalizeStructured(sb.String())
}

// stripPageHeaders removes repeating lines across pages (headers/footers)
// and "Página X de Y" page numbers. Pages are separated by \f (form feed).
func stripPageHeaders(text string) string {
	if text == "" {
		return ""
	}

	pages := strings.Split(text, "\f")
	if len(pages) <= 1 {
		// Single page — just strip page numbers.
		return strings.TrimSpace(pageNumberRe.ReplaceAllString(text, ""))
	}

	// Count line occurrences across pages to find repeating headers/footers.
	lineCounts := make(map[string]int)
	for _, page := range pages {
		seen := make(map[string]bool)
		for _, line := range strings.Split(page, "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			if !seen[trimmed] {
				seen[trimmed] = true
				lineCounts[trimmed]++
			}
		}
	}

	// Lines appearing on 2+ pages are headers/footers.
	repeating := make(map[string]bool)
	for line, count := range lineCounts {
		if count >= 2 {
			repeating[line] = true
		}
	}

	// Rebuild without repeating lines, page numbers, or form feeds.
	var result []string
	for _, page := range pages {
		var pageLines []string
		for line := range strings.SplitSeq(page, "\n") {
			trimmed := strings.TrimSpace(line)
			if repeating[trimmed] {
				continue
			}
			if pageNumberRe.MatchString(line) {
				continue
			}
			pageLines = append(pageLines, line)
		}
		cleaned := strings.TrimSpace(strings.Join(pageLines, "\n"))
		if cleaned != "" {
			result = append(result, cleaned)
		}
	}

	return strings.Join(result, "\n\n")
}

// isAllCaps returns true if the line contains only uppercase letters (including
// accented), digits, spaces, and punctuation — with at least one letter present.
func isAllCaps(line string) bool {
	if strings.TrimSpace(line) == "" {
		return false
	}

	hasLetter := false
	for _, r := range line {
		if unicode.IsLetter(r) {
			if unicode.IsLower(r) {
				return false
			}
			hasLetter = true
		}
	}
	return hasLetter
}

// isHeadingLine returns true if the line should be converted to a markdown heading.
// A line qualifies if it's ALL-CAPS and matches a known heading pattern.
// Pure length-based detection is intentionally NOT used — it causes false positives
// on professor names, department names, and table headers in guia docente PDFs.
func isHeadingLine(line string, patterns []string) bool {
	if !isAllCaps(line) {
		return false
	}

	// Strip accents for pattern matching.
	normalized := stripAccents(line)

	for _, pat := range patterns {
		if strings.Contains(normalized, pat) {
			return true
		}
	}

	return false
}

// toTitleCase converts an ALL-CAPS string to Title Case, lowercasing
// Spanish stop words (except the first word).
func toTitleCase(s string) string {
	if s == "" {
		return ""
	}

	words := strings.Fields(s)
	for i, w := range words {
		lower := strings.ToLower(w)
		if i > 0 && spanishStopWords[lower] {
			words[i] = lower
		} else {
			words[i] = titleWord(w)
		}
	}
	return strings.Join(words, " ")
}

// titleWord capitalizes the first rune and lowercases the rest.
func titleWord(w string) string {
	if w == "" {
		return ""
	}
	runes := []rune(strings.ToLower(w))
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// accentReplacer strips diacritical marks for pattern matching.
var accentReplacer = strings.NewReplacer(
	"Á", "A", "É", "E", "Í", "I", "Ó", "O", "Ú", "U", "Ñ", "N", "Ü", "U",
	"á", "a", "é", "e", "í", "i", "ó", "o", "ú", "u", "ñ", "n", "ü", "u",
)

// stripAccents removes diacritical marks for pattern matching.
func stripAccents(s string) string {
	return accentReplacer.Replace(s)
}
