package crawler

import (
	"bytes"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

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

	// Readability must not see hidden CMS blocks (the iDUMA session modal,
	// editor placeholders): they would win the length comparison below.
	stripBoilerplate(doc)

	readTitle, readText := extractReadable(doc, base)

	// Try content-area extraction (finds #main_content or similar)
	contentText := extractContentArea(doc)

	// Normalize once for comparison
	normContent := normalizeStructured(contentText)
	normRead := normalizeStructured(readText)

	// Use whichever yields more content
	text := normRead
	if len(normContent) > len(normRead) {
		text = normContent
	}

	// Final fallback to naive if both returned nothing
	if text == "" {
		text = normalizeStructured(extractNaive(doc))
	}

	return &HTMLResult{
		Title:    pageTitle(ogTitle, readTitle, findTitle(doc), base),
		Text:     text,
		Language: lang,
		Links:    links,
	}, nil
}

// pageTitle picks og:title (usually cleaner, no "X - Universidad de Málaga"
// suffix), else readability's title, else the document's <title>, skipping
// placeholders such as the "subject1" UMA's CMS stamps on course pages.
func pageTitle(ogTitle, readTitle, docTitle string, base *url.URL) string {
	title := docTitle
	for _, t := range []string{ogTitle, readTitle} {
		if !isPlaceholderTitle(t) {
			title = t
			break
		}
	}
	title = collapseSpace(title)

	// On a site's root page UMA's CMS sets <title> to "{site} - {og:title} - ...".
	// When og:title is a portal name shorter than the site name ("Grado UMA"
	// vs "Grado en INGENIERÍA INFORMÁTICA"), the site name is the title; a
	// longer og:title ("Escuela de Doctorado Universidad de Málaga" vs
	// "Doctorado") is already the specific one.
	if strings.Contains(strings.Trim(base.Path, "/"), "/") {
		return title
	}
	segments := strings.Split(collapseSpace(docTitle), " - ")
	site := segments[0]
	if utf8.RuneCountInString(site) <= utf8.RuneCountInString(title) {
		return title
	}
	for _, s := range segments[1:] {
		if s == title {
			return site
		}
	}
	return title
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

// placeholderTitle matches titles that authoring tools or the CMS stamp by
// default ("subject1", "Document1", "Microsoft Word - x.doc", ...); the page's
// own <title> or the URL-derived fallback is better than these.
var placeholderTitle = regexp.MustCompile(`(?i)^((subject|document|doc|untitled|title|slide|book|presentation|presentación|sin título|powerpoint presentation)[ _-]?\d*|microsoft word -.*|presentación de powerpoint.*)$`)

func isPlaceholderTitle(title string) bool {
	title = strings.TrimSpace(title)
	return title == "" || placeholderTitle.MatchString(title)
}

// extractOGTitle returns the first non-empty og:title meta content, or "".
func extractOGTitle(doc *html.Node) string {
	meta := findNode(doc, func(n *html.Node) bool {
		return n.Type == html.ElementNode && n.Data == "meta" &&
			strings.EqualFold(attrVal(n, "property"), "og:title") && attrVal(n, "content") != ""
	})
	if meta == nil {
		return ""
	}
	return attrVal(meta, "content")
}

// extractReadable uses go-readability to extract main article content.
func extractReadable(doc *html.Node, base *url.URL) (title, text string) {
	article, err := readability.FromDocument(doc, base)
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
// its text. This works better than readability for structured pages (tables,
// lists) that aren't articles.
func extractContentArea(doc *html.Node) string {
	contentNode := findContentNode(doc)
	if contentNode == nil {
		return ""
	}

	var sb strings.Builder
	extractText(contentNode, &sb)
	return sb.String()
}

// findTitle returns the document's own <title>: the first one in the HTML
// namespace, so inline SVG icon titles ("buscar") later in <body> never win.
func findTitle(doc *html.Node) string {
	n := findNode(doc, func(n *html.Node) bool {
		return n.Type == html.ElementNode && n.Data == "title" && n.Namespace == ""
	})
	if n == nil || n.FirstChild == nil || n.FirstChild.Type != html.TextNode {
		return ""
	}
	return strings.TrimSpace(n.FirstChild.Data)
}

// contentIDs are element IDs that mark a page's main content container.
var contentIDs = []string{"main_content", "content", "main-content", "page-content"}

// findContentNode returns the first element, in document order, that is a
// <main> or <article> or carries one of contentIDs.
func findContentNode(doc *html.Node) *html.Node {
	return findNode(doc, func(n *html.Node) bool {
		if n.Type != html.ElementNode {
			return false
		}
		if n.Data == "main" || n.Data == "article" {
			return true
		}
		id := attrVal(n, "id")
		for _, cid := range contentIDs {
			if strings.EqualFold(id, cid) {
				return true
			}
		}
		return false
	})
}

// headingLevel returns the heading level (1-6) for h1-h6, or 0 otherwise.
func headingLevel(tag string) int {
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

// blockElements are the elements a browser lays out on their own line, so
// their boundaries separate words even when the HTML has no whitespace there.
var blockElements = map[string]bool{
	"address": true, "article": true, "aside": true, "blockquote": true,
	"dd": true, "div": true, "dl": true, "dt": true, "figcaption": true,
	"figure": true, "h1": true, "h2": true, "h3": true, "h4": true, "h5": true,
	"h6": true, "hr": true, "li": true, "main": true, "ol": true, "p": true,
	"pre": true, "section": true, "table": true, "td": true, "th": true,
	"tr": true, "ul": true,
}

// nodeText returns the whitespace-collapsed text under an element, without
// markdown formatting.
func nodeText(n *html.Node) string {
	return collapseSpace(innerText(n))
}

// collapseSpace trims s and collapses every whitespace run in it to one space.
func collapseSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// innerText concatenates the text under n as is, with a space for every <br>
// and block boundary.
func innerText(n *html.Node) string {
	var sb strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		switch {
		case c.Type == html.TextNode:
			sb.WriteString(c.Data)
		case c.Type != html.ElementNode: // comments carry no text
		case c.Data == "br":
			sb.WriteByte(' ')
		case blockElements[c.Data]:
			sb.WriteString(" " + innerText(c) + " ")
		default:
			sb.WriteString(innerText(c))
		}
	}
	return sb.String()
}

// writeText appends s with every whitespace run collapsed to one space, and
// drops a space that would double another or start a line. Spacing between
// adjacent inline nodes is kept as the HTML has it.
func writeText(sb *strings.Builder, s string) {
	for _, r := range s {
		if !unicode.IsSpace(r) {
			sb.WriteRune(r)
			continue
		}
		if out := sb.String(); out != "" && !strings.HasSuffix(out, " ") && !strings.HasSuffix(out, "\n") {
			sb.WriteByte(' ')
		}
	}
}

// writeMarkup writes the words of raw between markdown markers; whitespace at
// raw's edges stays outside them so it still separates neighbouring words.
func writeMarkup(sb *strings.Builder, raw, open, close string) {
	text := collapseSpace(raw)
	if text == "" {
		writeText(sb, raw)
		return
	}
	if strings.TrimLeftFunc(raw, unicode.IsSpace) != raw {
		writeText(sb, " ")
	}
	sb.WriteString(open + text + close)
	if strings.TrimRightFunc(raw, unicode.IsSpace) != raw {
		writeText(sb, " ")
	}
}

// stripBoilerplate removes every boilerplate element, with its subtree, from n.
func stripBoilerplate(n *html.Node) {
	for c := n.FirstChild; c != nil; {
		next := c.NextSibling
		if isBoilerplate(c) {
			n.RemoveChild(c)
		} else {
			stripBoilerplate(c)
		}
		c = next
	}
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
		strings.Contains(cls, "blockcontainerplace") || strings.Contains(cls, "blockplace") ||
		strings.Contains(cls, "blockhiddentitle") {
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

// extractText recursively extracts visible text from a node as markdown.
func extractText(n *html.Node, sb *strings.Builder) {
	if n.Type == html.ElementNode {
		switch {
		case n.Data == "table":
			md := renderTableMarkdown(n)
			if md != "" {
				sb.WriteString("\n\n")
				sb.WriteString(md)
				sb.WriteString("\n\n")
			}
			return
		case headingLevel(n.Data) > 0:
			level := headingLevel(n.Data)
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
			writeText(sb, " ")
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
			writeMarkup(sb, innerText(n), "**", "**")
			return
		case n.Data == "em" || n.Data == "i":
			writeMarkup(sb, innerText(n), "*", "*")
			return
		case n.Data == "a":
			href := attrVal(n, "href")
			if href == "" || strings.HasPrefix(href, "#") || strings.HasPrefix(href, "javascript:") {
				writeText(sb, innerText(n))
			} else {
				writeMarkup(sb, innerText(n), "[", "]("+href+")")
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
		writeText(sb, n.Data)
	}

	block := n.Type == html.ElementNode && blockElements[n.Data]
	if block {
		writeText(sb, " ")
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		extractText(c, sb)
	}
	if block {
		writeText(sb, " ")
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
// Extraction runs against <body> only so <title>/<meta> in <head> never leak
// into the body text. Falls back to the whole document if <body> is missing
// (rare — html.Parse synthesizes it for any input).
func extractNaive(doc *html.Node) string {
	root := findFirstElement(doc, "body")
	if root == nil {
		root = doc
	}

	var sb strings.Builder
	extractText(root, &sb)
	return sb.String()
}

// findFirstElement returns the first descendant element with the given tag name.
func findFirstElement(doc *html.Node, tag string) *html.Node {
	return findNode(doc, func(n *html.Node) bool {
		return n.Type == html.ElementNode && n.Data == tag
	})
}

// findNode returns the first descendant of doc, in document order, that matches.
func findNode(doc *html.Node, match func(*html.Node) bool) *html.Node {
	for n := range doc.Descendants() {
		if match(n) {
			return n
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
					cells = append(cells, nodeText(c))
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

// normalizeURL lowercases the host, puts uma.es hosts on https, removes
// query-string parameters whose names appear in stripParams (case-insensitive)
// and canonicalises the remaining query by sorting keys. Dedup-friendly: two
// URLs that differ only in host case, http vs https on UMA (which redirects
// http to https), tracking parameters or key order normalise to the same string.
//
// Returns the raw input unchanged on parse failure so caller flow isn't
// disrupted by malformed URLs that slip through resolveLink.
func normalizeURL(rawURL string, stripParams []string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	u.Host = strings.ToLower(u.Host)
	if host := u.Hostname(); host == "uma.es" || strings.HasSuffix(host, ".uma.es") {
		u.Scheme = "https"
	}
	if u.RawQuery == "" {
		return u.String()
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
