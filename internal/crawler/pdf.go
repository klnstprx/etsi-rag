package crawler

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"etsi-rag/internal/config"
	"etsi-rag/internal/markers"
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
			if isPlaceholderPDFTitle(title) {
				return ""
			}
			return title
		}
	}
	return ""
}

// placeholderPDFTitle matches metadata titles that authoring tools stamp by
// default ("subject1", "Document1", "PowerPoint Presentation", ...). These are
// worse than the URL-derived fallback, so extractPDFTitle treats them as empty.
var placeholderPDFTitle = regexp.MustCompile(`(?i)^(subject|document|doc|untitled|title|slide|book|presentation|presentación|sin título|microsoft word|powerpoint presentation)[ _-]?\d*\b`)

func isPlaceholderPDFTitle(title string) bool {
	return strings.TrimSpace(title) == "" || placeholderPDFTitle.MatchString(strings.TrimSpace(title))
}

// pdfTextToMarkdown converts raw pdftotext output to structured markdown with
// a single `[[image:pN]]` line emitted at the top of each page block. The
// chunker treats markers as page boundaries (see markers.ExtractBoundaries),
// stamps each emitted chunk with the refs it covers, and strips markers from
// final chunk content — so markers are an internal parser→chunker protocol,
// never user-facing.
//
// Pipeline: strip page headers → emit page marker → detect headings → normalize.
func pdfTextToMarkdown(raw string, cfg config.PDFConfig) string {
	if raw == "" {
		return ""
	}

	pages := stripPageHeaders(raw)
	if len(pages) == 0 {
		return ""
	}

	var sb strings.Builder
	for i, page := range pages {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(markers.Format(fmt.Sprintf("p%d", i+1)))
		sb.WriteByte('\n')
		for line := range strings.SplitSeq(page, "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				sb.WriteByte('\n')
				continue
			}
			if isHeadingLine(trimmed, cfg.HeadingPatterns) {
				sb.WriteString("## ")
				sb.WriteString(toTitleCase(trimmed))
				sb.WriteByte('\n')
				continue
			}
			sb.WriteString(line)
			sb.WriteByte('\n')
		}
	}

	return normalizeStructured(sb.String())
}

// stripPageHeaders splits raw pdftotext output on \f (form feed) page
// boundaries, removes lines that repeat across pages (running headers and
// footers) and "Página X de Y" page-number lines, and returns the cleaned
// per-page text. The slice length equals the number of pages with at least
// one non-empty line; callers can map slice index → page number.
func stripPageHeaders(text string) []string {
	if text == "" {
		return nil
	}

	pages := strings.Split(text, "\f")

	// Count line occurrences across pages to find repeating headers/footers.
	// Single-page inputs have no repeating set; this loop is a no-op for them.
	repeating := make(map[string]bool)
	if len(pages) > 1 {
		lineCounts := make(map[string]int)
		for _, page := range pages {
			seen := make(map[string]bool)
			for line := range strings.SplitSeq(page, "\n") {
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
		for line, count := range lineCounts {
			if count >= 2 {
				repeating[line] = true
			}
		}
	}

	out := make([]string, 0, len(pages))
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
			out = append(out, cleaned)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// ExtractPDFImages renders each PDF page to a PNG using pdftoppm (poppler) at
// the given DPI. The returned slice is ordered by page number — pages[0] is
// page 1, pages[1] is page 2, etc. Pages that fail to render individually are
// skipped (returned slice may have gaps); a fully empty result returns
// (nil, error).
func ExtractPDFImages(ctx context.Context, pdfData []byte, dpi int) ([][]byte, error) {
	if dpi <= 0 {
		return nil, nil
	}

	tmp, err := os.CreateTemp("", "crawl-img-*.pdf")
	if err != nil {
		return nil, fmt.Errorf("creating temp pdf: %w", err)
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(pdfData); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("writing temp pdf: %w", err)
	}
	tmp.Close()

	outDir, err := os.MkdirTemp("", "crawl-img-out-*")
	if err != nil {
		return nil, fmt.Errorf("creating output dir: %w", err)
	}
	defer os.RemoveAll(outDir)

	prefix := filepath.Join(outDir, "page")
	cmd := exec.CommandContext(ctx, "pdftoppm",
		"-png", "-r", fmt.Sprintf("%d", dpi),
		tmp.Name(), prefix,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("pdftoppm: %w: %s", err, stderr.String())
	}

	entries, err := os.ReadDir(outDir)
	if err != nil {
		return nil, fmt.Errorf("reading output dir: %w", err)
	}

	// pdftoppm emits page-1.png, page-2.png, ... (no zero-padding by default).
	// Sort by name to ensure page order is preserved across filesystems that
	// don't guarantee directory iteration order.
	type entry struct {
		page int
		name string
	}
	var files []entry
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".png") {
			continue
		}
		// Strip "page-" prefix and ".png" suffix to get the page number.
		base := strings.TrimSuffix(e.Name(), ".png")
		base = strings.TrimPrefix(base, "page-")
		var n int
		if _, err := fmt.Sscanf(base, "%d", &n); err != nil {
			continue
		}
		files = append(files, entry{page: n, name: e.Name()})
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("pdftoppm produced no output")
	}
	sort.Slice(files, func(i, j int) bool { return files[i].page < files[j].page })

	images := make([][]byte, 0, len(files))
	for _, f := range files {
		data, err := os.ReadFile(filepath.Join(outDir, f.name))
		if err != nil {
			// Per plan: log+skip per-page failures, don't fail the whole doc.
			continue
		}
		images = append(images, data)
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("pdftoppm produced output but all reads failed")
	}
	return images, nil
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
