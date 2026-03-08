package crawler

import (
	"bytes"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	readability "codeberg.org/readeck/go-readability/v2"
	"golang.org/x/net/html"
)

var multiNewlineRe = regexp.MustCompile(`\n{3,}`)

// HTMLResult contains the extracted content from an HTML page.
type HTMLResult struct {
	Title       string
	Description string // from og:description or <meta name="description">
	Text        string
	Links       []string // absolute URLs
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

	// Extract OG metadata
	ogTitle, ogDesc := extractOGMeta(doc)

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
		Title:       title,
		Description: ogDesc,
		Text:        text,
		Links:       links,
	}, nil
}

// extractOGMeta extracts og:title and og:description (or fallback meta description) from a parsed DOM.
func extractOGMeta(doc *html.Node) (title, description string) {
	var metaDesc string
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "meta" {
			prop := attrVal(n, "property")
			name := attrVal(n, "name")
			content := attrVal(n, "content")
			switch {
			case strings.EqualFold(prop, "og:title"):
				title = content
			case strings.EqualFold(prop, "og:description"):
				description = content
			case strings.EqualFold(name, "description") && metaDesc == "":
				metaDesc = content
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	if description == "" {
		description = metaDesc
	}
	return title, description
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
func extractNaive(doc *html.Node) (title, text string) {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style", "nav", "header", "footer", "noscript":
				return
			case "title":
				if n.FirstChild != nil && n.FirstChild.Type == html.TextNode {
					title = strings.TrimSpace(n.FirstChild.Data)
				}
			}
			// Delegate to extractText for markdown-aware rendering
			extractText(n, &sb)
			return
		}

		if n.Type == html.TextNode {
			t := strings.TrimSpace(n.Data)
			if t != "" {
				sb.WriteString(t)
				sb.WriteByte(' ')
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(doc)
	return title, sb.String()
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
