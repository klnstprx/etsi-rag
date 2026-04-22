package crawler

import (
	"bytes"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"etsi-rag/internal/domain"

	readability "codeberg.org/readeck/go-readability/v2"
	"golang.org/x/net/html"
)

var multiNewlineRe = regexp.MustCompile(`\n{3,}`)

// HTMLResult contains the extracted content from an HTML page.
type HTMLResult struct {
	Title    string
	Text     string
	Language string   // normalized BCP-47 primary subtag ("es", "en") or "" if absent/unsupported
	Links    []string // absolute URLs
}

// ParseHTML extracts title, main content text, and links from raw HTML.
// Tries readability first, then compares with content-area extraction.
// Uses whichever yields more text. Links are always extracted from the
// full DOM (including nav/sidebar) for BFS crawling.
func ParseHTML(body []byte, baseURL string) (*HTMLResult, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}

	// Parse DOM once and reuse across extraction functions
	doc, err := html.Parse(bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("parsing HTML: %w", err)
	}

	// Extract links from full DOM (need nav, sidebar, footer links for BFS)
	links := extractLinks(doc, base)

	// Extract OG title + <html lang>
	ogTitle := extractOGTitle(doc)
	lang := extractHTMLLang(doc)

	// Try readability (needs its own parse — library API)
	title, readText := extractReadable(body, base)

	// Try content-area extraction (finds #main_content or similar, skips boilerplate)
	contentTitle, contentText := extractContentArea(doc)

	// Normalize once for comparison
	normContent := normalizeStructured(contentText)
	normRead := normalizeStructured(readText)

	// Use whichever yields more content
	text := normRead
	if len(normContent) > len(normRead) {
		text = normContent
		if title == "" {
			title = contentTitle
		}
	}

	// Final fallback to naive if both returned nothing
	if text == "" {
		title, text = extractNaive(doc)
		text = normalizeStructured(text)
	}

	// OG title is usually cleaner (no "X - Universidad de Málaga" suffix)
	if ogTitle != "" {
		title = ogTitle
	}

	return &HTMLResult{
		Title:    title,
		Text:     text,
		Language: lang,
		Links:    links,
	}, nil
}

// extractHTMLLang reads the `lang` attribute from the first <html> element
// and returns the normalised primary subtag if it names a language supported
// by domain.Language. Empty string means "fall back to the source language".
//
// Normalisation:
//   - trim whitespace
//   - lowercase
//   - drop the region subtag (en-US → en, es-ES → es)
//   - reject anything not in the domain.Language.Valid() set
func extractHTMLLang(doc *html.Node) string {
	htmlNode := findFirstElement(doc, "html")
	if htmlNode == nil {
		return ""
	}
	raw := strings.TrimSpace(attrVal(htmlNode, "lang"))
	if raw == "" {
		return ""
	}
	primary := strings.ToLower(raw)
	if i := strings.Index(primary, "-"); i > 0 {
		primary = primary[:i]
	}
	if !domain.Language(primary).Valid() {
		return ""
	}
	return primary
}

// extractOGTitle returns the og:title meta tag content, or empty string if absent.
func extractOGTitle(doc *html.Node) string {
	var title string
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if title != "" {
			return
		}
		if n.Type == html.ElementNode && n.Data == "meta" {
			if strings.EqualFold(attrVal(n, "property"), "og:title") {
				title = attrVal(n, "content")
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return title
}

// extractReadable uses go-readability to extract main article content.
func extractReadable(body []byte, base *url.URL) (title, text string) {
	article, err := readability.FromReader(bytes.NewReader(body), base)
	if err != nil {
		return "", ""
	}

	title = article.Title()

	if article.Node != nil {
		var buf bytes.Buffer
		if err := article.RenderText(&buf); err == nil {
			text = buf.String()
		}
	}

	return title, text
}

// extractContentArea finds a main content container in the DOM and extracts
// text from it, skipping nav/header/footer/modal elements. This works better
// than readability for structured pages (tables, lists) that aren't articles.
func extractContentArea(doc *html.Node) (title, text string) {
	// Find title
	title = findTitle(doc)

	// Look for a main content container by ID or tag
	contentNode := findContentNode(doc)
	if contentNode == nil {
		return title, ""
	}

	// Extract text from the content node, skipping boilerplate children
	var sb strings.Builder
	extractText(contentNode, &sb)
	return title, sb.String()
}

// findTitle extracts the <title> text.
func findTitle(doc *html.Node) string {
	var title string
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "title" {
			if n.FirstChild != nil && n.FirstChild.Type == html.TextNode {
				title = strings.TrimSpace(n.FirstChild.Data)
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return title
}

// findContentNode searches for a main content container by ID or semantic tag.
func findContentNode(doc *html.Node) *html.Node {
	// Priority list of content container IDs
	contentIDs := []string{"main_content", "content", "main-content", "page-content"}

	var found *html.Node
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if found != nil {
			return
		}
		if n.Type == html.ElementNode {
			id := attrVal(n, "id")
			for _, cid := range contentIDs {
				if strings.EqualFold(id, cid) {
					found = n
					return
				}
			}
			// Also check <main> and <article> tags
			if n.Data == "main" || n.Data == "article" {
				found = n
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return found
}

// isHeading returns the heading level (1-6) if the node is h1-h6, or 0 otherwise.
func isHeading(tag string) int {
	switch tag {
	case "h1":
		return 1
	case "h2":
		return 2
	case "h3":
		return 3
	case "h4":
		return 4
	case "h5":
		return 5
	case "h6":
		return 6
	}
	return 0
}

// nodeText extracts plain text from a node without any markdown formatting.
func nodeText(n *html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return strings.Join(strings.Fields(sb.String()), " ")
}

// isBoilerplate returns true if the element should be skipped.
func isBoilerplate(n *html.Node) bool {
	if n.Type != html.ElementNode {
		return false
	}
	switch n.Data {
	case "script", "style", "noscript", "nav", "header", "footer":
		return true
	}
	cls := strings.ToLower(attrVal(n, "class"))
	id := strings.ToLower(attrVal(n, "id"))
	if strings.Contains(cls, "modal") || strings.Contains(cls, "cookie") ||
		strings.Contains(cls, "overlay") {
		return true
	}
	if strings.Contains(id, "modal") || strings.Contains(id, "cookie") {
		return true
	}
	if strings.Contains(cls, "beforecontent") || strings.Contains(cls, "aftercontent") ||
		strings.Contains(cls, "blockcontainerplace") || strings.Contains(cls, "blockplace") {
		return true
	}
	if strings.HasPrefix(id, "beforecontent") || strings.HasPrefix(id, "aftercontent") {
		return true
	}
	return false
}

// parentTag returns the tag name of a node's parent element, or empty string.
func parentTag(n *html.Node) string {
	if n.Parent != nil && n.Parent.Type == html.ElementNode {
		return n.Parent.Data
	}
	return ""
}

// extractText recursively extracts visible text from a node as markdown, skipping boilerplate.
func extractText(n *html.Node, sb *strings.Builder) {
	if n.Type == html.ElementNode {
		if isBoilerplate(n) {
			return
		}

		switch {
		case n.Data == "table":
			md := renderTableMarkdown(n)
			if md != "" {
				sb.WriteString("\n\n")
				sb.WriteString(md)
				sb.WriteString("\n\n")
			}
			return
		case isHeading(n.Data) > 0:
			level := isHeading(n.Data)
			text := nodeText(n)
			if text != "" {
				sb.WriteString("\n\n")
				sb.WriteString(strings.Repeat("#", level))
				sb.WriteByte(' ')
				sb.WriteString(text)
				sb.WriteString("\n\n")
			}
			return
		case n.Data == "p":
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				extractText(c, sb)
			}
			sb.WriteString("\n\n")
			return
		case n.Data == "li":
			parent := parentTag(n)
			if parent == "ol" {
				sb.WriteString("\n1. ")
			} else {
				sb.WriteString("\n- ")
			}
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				extractText(c, sb)
			}
			return
		case n.Data == "br":
			sb.WriteByte('\n')
			return
		case n.Data == "strong" || n.Data == "b":
			text := nodeText(n)
			if text != "" {
				sb.WriteString("**")
				sb.WriteString(text)
				sb.WriteString("**")
			}
			return
		case n.Data == "em" || n.Data == "i":
			text := nodeText(n)
			if text != "" {
				sb.WriteByte('*')
				sb.WriteString(text)
				sb.WriteByte('*')
			}
			return
		case n.Data == "a":
			href := attrVal(n, "href")
			text := nodeText(n)
			if text != "" && href != "" && !strings.HasPrefix(href, "#") &&
				!strings.HasPrefix(href, "javascript:") {
				sb.WriteString("[")
				sb.WriteString(text)
				sb.WriteString("](")
				sb.WriteString(href)
				sb.WriteString(")")
			} else if text != "" {
				sb.WriteString(text)
			}
			return
		case n.Data == "ul" || n.Data == "ol":
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				extractText(c, sb)
			}
			sb.WriteByte('\n')
			return
		}
	}

	if n.Type == html.TextNode {
		t := strings.TrimSpace(n.Data)
		if t != "" {
			sb.WriteString(t)
			sb.WriteByte(' ')
		}
	}

	for c := n.FirstChild; c != nil; c = c.NextSibling {
		extractText(c, sb)
	}
}

// attrVal returns the value of an HTML attribute, or empty string if not found.
func attrVal(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if a.Key == key {
			return a.Val
		}
	}
	return ""
}

// extractNaive walks the full DOM and concatenates visible text as markdown.
// Used as final fallback when both readability and content-area extraction fail.
//
// Title is captured separately via findTitle; body extraction runs against
// <body> only so <title>/<meta>/<script>/<style> in <head> never leak into
// the body text. Falls back to the whole document if <body> is missing
// (rare — html.Parse synthesizes it for any input).
func extractNaive(doc *html.Node) (title, text string) {
	title = findTitle(doc)

	root := findFirstElement(doc, "body")
	if root == nil {
		root = doc
	}

	var sb strings.Builder
	extractText(root, &sb)
	return title, sb.String()
}

// findFirstElement returns the first descendant ElementNode with the given tag name.
func findFirstElement(n *html.Node, tag string) *html.Node {
	if n == nil {
		return nil
	}
	if n.Type == html.ElementNode && n.Data == tag {
		return n
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if found := findFirstElement(c, tag); found != nil {
			return found
		}
	}
	return nil
}

// renderTableMarkdown converts an HTML <table> node to markdown table format.
func renderTableMarkdown(tableNode *html.Node) string {
	var rows [][]string

	// Collect rows from <thead>, <tbody>, or direct <tr> children.
	var walkRows func(*html.Node)
	walkRows = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "tr" {
			var cells []string
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && (c.Data == "td" || c.Data == "th") {
					cells = append(cells, cellText(c))
				}
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
			return
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walkRows(c)
		}
	}
	walkRows(tableNode)

	if len(rows) == 0 {
		return ""
	}

	// Determine max columns
	maxCols := 0
	for _, row := range rows {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}

	// Pad rows to maxCols
	for i := range rows {
		for len(rows[i]) < maxCols {
			rows[i] = append(rows[i], "")
		}
	}

	var sb strings.Builder

	// Write header row
	sb.WriteString("| ")
	sb.WriteString(strings.Join(rows[0], " | "))
	sb.WriteString(" |\n")

	// Separator
	sep := make([]string, maxCols)
	for i := range sep {
		sep[i] = "---"
	}
	sb.WriteString("| ")
	sb.WriteString(strings.Join(sep, " | "))
	sb.WriteString(" |\n")

	// Data rows (skip first — already used as header)
	for _, row := range rows[1:] {
		sb.WriteString("| ")
		sb.WriteString(strings.Join(row, " | "))
		sb.WriteString(" |\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

// cellText extracts plain text from a <td> or <th> node, normalizing whitespace.
func cellText(n *html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return strings.Join(strings.Fields(sb.String()), " ")
}

// extractLinks walks the full DOM and extracts all <a href> links.
// Includes links from nav, sidebar, footer — needed for BFS crawling.
func extractLinks(doc *html.Node, base *url.URL) []string {
	seen := make(map[string]bool)
	var links []string

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			for _, attr := range n.Attr {
				if attr.Key == "href" {
					link := resolveLink(base, attr.Val)
					if link != "" && !seen[link] {
						seen[link] = true
						links = append(links, link)
					}
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(doc)
	return links
}

// normalizeURL removes query-string parameters whose names appear in
// stripParams (case-insensitive) and canonicalises the remaining query by
// sorting keys. Dedup-friendly: two URLs that differ only in tracking
// parameters or key order normalise to the same string.
//
// Returns the raw input unchanged on parse failure so caller flow isn't
// disrupted by malformed URLs that slip through resolveLink.
func normalizeURL(rawURL string, stripParams []string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	if u.RawQuery == "" {
		return rawURL
	}

	strip := make(map[string]bool, len(stripParams))
	for _, p := range stripParams {
		strip[strings.ToLower(p)] = true
	}

	q := u.Query()
	for key := range q {
		if strip[strings.ToLower(key)] {
			q.Del(key)
		}
	}
	u.RawQuery = q.Encode() // Encode() sorts keys alphabetically
	return u.String()
}

// mustParseURL parses a URL or panics. For use in tests.
func mustParseURL(rawURL string) *url.URL {
	u, err := url.Parse(rawURL)
	if err != nil {
		panic(err)
	}
	return u
}

// resolveLink resolves a relative URL against a base URL.
// Returns empty string for invalid or non-HTTP links.
func resolveLink(base *url.URL, href string) string {
	href = strings.TrimSpace(href)

	// Skip non-HTTP schemes and fragments
	if href == "" || strings.HasPrefix(href, "#") ||
		strings.HasPrefix(href, "mailto:") ||
		strings.HasPrefix(href, "javascript:") ||
		strings.HasPrefix(href, "tel:") {
		return ""
	}

	ref, err := url.Parse(href)
	if err != nil {
		return ""
	}

	resolved := base.ResolveReference(ref)
	// Strip fragment
	resolved.Fragment = ""

	if resolved.Scheme != "http" && resolved.Scheme != "https" {
		return ""
	}

	return resolved.String()
}

// normalizeWhitespace collapses runs of whitespace into single spaces
// and trims the result.
func normalizeWhitespace(s string) string {
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}

// normalizeStructured collapses 3+ consecutive newlines to 2, trims trailing
// spaces per line, and preserves paragraph boundaries (\n\n).
func normalizeStructured(s string) string {
	// Split into lines, trim trailing spaces
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	result := strings.Join(lines, "\n")

	// Collapse 3+ newlines to 2
	result = multiNewlineRe.ReplaceAllString(result, "\n\n")

	return strings.TrimSpace(result)
}
