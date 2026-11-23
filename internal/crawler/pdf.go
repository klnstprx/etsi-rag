package crawler

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

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
	path, err := writeTempPDF(pdfData)
	if err != nil {
		return nil, err
	}
	defer os.Remove(path)

	// Run pdftotext in default (reading order) mode — no -layout flag.
	// Default mode undoes column formatting and reads in logical order.
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "pdftotext", "-enc", "UTF-8", path, "-")
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("pdftotext: %w: %s", err, stderr.String())
	}

	raw := stdout.String()
	title := extractPDFTitle(ctx, path)
	if title == "" {
		title = subjectGuideTitle(raw)
	}
	text := pdfTextToMarkdown(raw, cfg)

	return &PDFResult{Title: title, Text: text}, nil
}

// writeTempPDF writes data to a temp file for the poppler CLIs, which only
// read from disk. The caller removes the returned path.
func writeTempPDF(data []byte) (string, error) {
	tmp, err := os.CreateTemp("", "crawl-*.pdf")
	if err != nil {
		return "", fmt.Errorf("creating temp pdf: %w", err)
	}
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if err := errors.Join(werr, cerr); err != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("writing temp pdf: %w", err)
	}
	return tmp.Name(), nil
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
		title, ok := strings.CutPrefix(line, "Title:")
		if !ok {
			continue
		}
		if title = strings.TrimSpace(title); isPlaceholderTitle(title) || isJunkPDFTitle(title) {
			return ""
		}
		return title
	}
	return ""
}

// Bounds on a usable pdfinfo Title. Authoring tools fill Title with the first
// line of the source document, cut at about 126 characters on UMA's PDFs.
const (
	minPDFTitleLetters = 4
	maxPDFTitleRunes   = 120
)

// authoringFileTitle matches a Title that is the source file's name
// ("Listado definitivo.xlsx").
var authoringFileTitle = regexp.MustCompile(`(?i)\.(docx?|xlsx?|xlsm|rtf|odt|pptx?|pdf|html?)$`)

// cutTitle matches a Title ending in a one- or two-letter word, which on UMA's
// PDFs is a template line cut short ("ESCUELA TÉCNICA SUPERIOR DE");
// romanNumeral spares course numbers ("Cálculo II").
var (
	cutTitle     = regexp.MustCompile(`(^|\PL)\pL{1,2}$`)
	romanNumeral = regexp.MustCompile(`\s[IVX]+$`)
)

// isJunkPDFTitle reports whether a pdfinfo Title is an authoring leftover
// rather than the document's name: a source filename, a fragment ("D"), a
// form label ("Destinatario:") or a first line cut mid-phrase.
func isJunkPDFTitle(title string) bool {
	letters := 0
	for _, r := range title {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	return letters < minPDFTitleLetters ||
		utf8.RuneCountInString(title) > maxPDFTitleRunes ||
		strings.HasSuffix(title, ":") ||
		authoringFileTitle.MatchString(title) ||
		(cutTitle.MatchString(title) && !romanNumeral.MatchString(title))
}

// subjectGuideTitle names a sara.uma.es subject guide from the "Asignatura:"
// and "Grado/Máster en:" fields of its first page, because these PDFs carry no
// Title. It returns "" for any other document.
func subjectGuideTitle(raw string) string {
	firstPage, _, _ := strings.Cut(raw, "\f")
	degree, isGuide := fieldValue(firstPage, "Grado/Máster en:")
	subject, _ := fieldValue(firstPage, "Asignatura:")
	if !isGuide || subject == "" {
		return ""
	}
	if degree == "" {
		return "Guía docente de " + subject
	}
	return fmt.Sprintf("Guía docente de %s (%s)", subject, degree)
}

// fieldValue finds the line holding label and returns the next non-empty line
// as its value. pdftotext's reading order sometimes puts another label there,
// so a value ending in ":" counts as missing.
func fieldValue(page, label string) (value string, found bool) {
	lines := strings.Split(page, "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) != label {
			continue
		}
		for _, next := range lines[i+1:] {
			if next = strings.TrimSpace(next); next != "" {
				if strings.HasSuffix(next, ":") {
					return "", true
				}
				return next, true
			}
		}
		return "", true
	}
	return "", false
}

// pdfTextToMarkdown converts raw pdftotext output to structured markdown with
// a single `[[image:pN]]` line emitted at the top of each page block, N being
// the physical page number so it matches the page's render; pages left
// without text emit nothing. The chunker treats markers as page boundaries
// (see markers.ExtractBoundaries), stamps each emitted chunk with the refs it covers, and strips markers from
// final chunk content — so markers are an internal parser→chunker protocol,
// never user-facing.
//
// Pipeline: strip page headers → emit page marker → detect headings → normalize.
func pdfTextToMarkdown(raw string, cfg config.PDFConfig) string {
	if raw == "" {
		return ""
	}

	var sb strings.Builder
	for i, page := range stripPageHeaders(raw) {
		if page == "" {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(markers.Format(pageRef(i + 1)))
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

// A running header or footer is a line found once on, and within
// headerZoneLines non-empty lines of the top or bottom of, at least
// max(minHeaderPages, headerPageShare of all) pages. Fewer pages cannot tell a
// header from a table cell repeated by chance, and a line repeated within a
// page is a cell ("6", "Optativa"), not a header.
const (
	headerZoneLines = 6 // fits the UMA e-signature footer; 4 leaves half of it
	minHeaderPages  = 3
	headerPageShare = 0.6
)

// stripPageHeaders splits raw pdftotext output on \f (form feed) page
// boundaries and returns each page's text without its running header and
// footer or "Página X de Y" lines. The slice is indexed by physical page and
// holds "" for a page with no text left. Running lines are dropped only near
// the edges of a page (see headerLines), so the same text deeper in its body
// (a table cell) stays.
func stripPageHeaders(text string) []string {
	if text == "" {
		return nil
	}
	pages := strings.Split(strings.TrimSuffix(text, "\f"), "\f")
	for i, page := range pages {
		// On the whole page, so a "Página X de" split from its "Y" goes too.
		pages[i] = pageNumberRe.ReplaceAllString(page, "")
	}
	running := runningLines(pages)

	out := make([]string, len(pages))
	for i, page := range pages {
		lines := strings.Split(page, "\n")
		header := headerLines(lines, running)
		var body []string
		for j, line := range lines {
			if !header[j] {
				body = append(body, line)
			}
		}
		out[i] = strings.TrimSpace(strings.Join(body, "\n"))
	}
	return out
}

// headerLines marks the lines of a page that belong to its running header or
// footer. On a long page these are the running lines in its edge zone, even
// with a body line between them and the edge (pdftotext often reads a footer
// before the page's last body line), and none deeper in. On a short page the
// zone is the whole page, so only the running and blank lines touching an
// edge are marked, and a repeated cell mid-page stays.
func headerLines(lines []string, running map[string]bool) []bool {
	header := make([]bool, len(lines))
	zone, whole := edgeZone(lines)
	if !whole {
		for _, j := range zone {
			header[j] = running[strings.TrimSpace(lines[j])]
		}
		return header
	}
	isEdge := func(line string) bool {
		trimmed := strings.TrimSpace(line)
		return trimmed == "" || running[trimmed]
	}
	for j := 0; j < len(lines) && isEdge(lines[j]); j++ {
		header[j] = true
	}
	for j := len(lines) - 1; j >= 0 && isEdge(lines[j]); j-- {
		header[j] = true
	}
	return header
}

// edgeZone returns the indexes of a page's first and last headerZoneLines
// non-empty lines, and whether those are all of its non-empty lines.
func edgeZone(lines []string) (zone []int, whole bool) {
	for j, line := range lines {
		if strings.TrimSpace(line) != "" {
			zone = append(zone, j)
		}
	}
	if len(zone) <= 2*headerZoneLines {
		return zone, true
	}
	return slices.Concat(zone[:headerZoneLines], zone[len(zone)-headerZoneLines:]), false
}

// runningLines returns the trimmed lines that recur, once per page, in the
// edge zone of enough pages to be a running header or footer.
func runningLines(pages []string) map[string]bool {
	if len(pages) < minHeaderPages {
		return nil
	}
	counts := make(map[string]int)
	for _, page := range pages {
		lines := strings.Split(page, "\n")
		onPage := make(map[string]int)
		for _, line := range lines {
			onPage[strings.TrimSpace(line)]++
		}
		zone, _ := edgeZone(lines)
		for _, j := range zone {
			if line := strings.TrimSpace(lines[j]); onPage[line] == 1 {
				counts[line]++
			}
		}
	}
	minPages := max(minHeaderPages, int(math.Ceil(headerPageShare*float64(len(pages)))))
	running := make(map[string]bool)
	for line, n := range counts {
		if n >= minPages {
			running[line] = true
		}
	}
	return running
}

// pageRef is the artifact ref, and marker, of a PDF page.
func pageRef(page int) string {
	return fmt.Sprintf("p%d", page)
}

// PageImage is one rendered PDF page; Ref matches the page's text marker.
type PageImage struct {
	Ref string
	PNG []byte
}

// ExtractPDFImages renders each PDF page to a PNG using pdftoppm (poppler) at
// the given DPI, ordered by page number. Pages whose PNG can't be read are
// skipped; each image keeps its own page's ref, so a gap does not shift the
// pages after it. A fully empty result is an error.
func ExtractPDFImages(ctx context.Context, pdfData []byte, dpi int) ([]PageImage, error) {
	path, err := writeTempPDF(pdfData)
	if err != nil {
		return nil, err
	}
	defer os.Remove(path)

	outDir, err := os.MkdirTemp("", "crawl-img-out-*")
	if err != nil {
		return nil, fmt.Errorf("creating output dir: %w", err)
	}
	defer os.RemoveAll(outDir)

	prefix := filepath.Join(outDir, "page")
	cmd := exec.CommandContext(ctx, "pdftoppm",
		"-png", "-r", fmt.Sprintf("%d", dpi),
		path, prefix,
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

	images := make([]PageImage, 0, len(files))
	for _, f := range files {
		data, err := os.ReadFile(filepath.Join(outDir, f.name))
		if err != nil {
			continue
		}
		images = append(images, PageImage{Ref: pageRef(f.page), PNG: data})
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
