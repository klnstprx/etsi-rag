package crawler

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/net/html"
)

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("failed to load fixture %s: %v", name, err)
	}
	return data
}

func TestParseHTML_ExtractsMainContent(t *testing.T) {
	body := loadFixture(t, "article.html")
	result, err := ParseHTML(body, "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should contain main article text
	if !strings.Contains(result.Text, "diseñar, desarrollar y mantener sistemas informáticos") {
		t.Error("expected main article content in text")
	}
	if !strings.Contains(result.Text, "plan de estudios incluye asignaturas") {
		t.Error("expected second paragraph in text")
	}

	// Should NOT contain nav/header/footer/sidebar boilerplate
	if strings.Contains(result.Text, "Footer content") {
		t.Error("text should not contain footer content")
	}
	if strings.Contains(result.Text, "analytics") {
		t.Error("text should not contain script content")
	}
}

func TestParseHTML_ExtractsTitle(t *testing.T) {
	body := loadFixture(t, "article.html")
	result, err := ParseHTML(body, "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Title == "" {
		t.Fatal("expected non-empty title")
	}
	if !strings.Contains(result.Title, "Informática") {
		t.Errorf("expected title to contain 'Informática', got %q", result.Title)
	}
}

func TestParseHTML_ExtractsAllLinks(t *testing.T) {
	body := loadFixture(t, "article.html")
	result, err := ParseHTML(body, "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Links should include nav, sidebar, footer, and main content links
	// (we need all links for BFS crawling, even from boilerplate areas)
	wantLinks := []string{
		"https://www.uma.es/etsi-informatica/",
		"https://www.uma.es/grado-en-ingenieria-del-software/",
		"https://www.uma.es/home",
		"https://www.uma.es/grado-en-ingenieria-informatica/cms/menu/plan-de-estudios/",
		"https://www.uma.es/media/files/memoria_verificacion.pdf",
		"https://www.uma.es/noticias/",
		"https://www.uma.es/legal",
	}

	linkSet := make(map[string]bool)
	for _, l := range result.Links {
		linkSet[l] = true
	}

	for _, want := range wantLinks {
		if !linkSet[want] {
			t.Errorf("expected link %q in results, got links: %v", want, result.Links)
		}
	}
}

func TestParseHTML_ResolvesRelativeLinks(t *testing.T) {
	body := loadFixture(t, "article.html")
	result, err := ParseHTML(body, "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	linkSet := make(map[string]bool)
	for _, l := range result.Links {
		linkSet[l] = true
	}

	// relative/page.html should resolve against base URL
	want := "https://www.uma.es/grado-en-ingenieria-informatica/relative/page.html"
	if !linkSet[want] {
		t.Errorf("expected resolved relative link %q, got links: %v", want, result.Links)
	}
}

func TestParseHTML_FiltersNonHTTPLinks(t *testing.T) {
	body := loadFixture(t, "article.html")
	result, err := ParseHTML(body, "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, link := range result.Links {
		if strings.HasPrefix(link, "mailto:") ||
			strings.HasPrefix(link, "javascript:") ||
			strings.HasPrefix(link, "tel:") ||
			strings.HasPrefix(link, "#") {
			t.Errorf("should not contain non-HTTP link: %s", link)
		}
	}
}

func TestParseHTML_NormalizesWhitespace(t *testing.T) {
	html := []byte(`<html><head><title>Test</title></head><body>
		<p>  Multiple   spaces   and
		newlines  here  </p>
	</body></html>`)

	result, err := ParseHTML(html, "https://example.com/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if strings.Contains(result.Text, "  ") {
		t.Errorf("text should not contain consecutive spaces: %q", result.Text)
	}
	if strings.HasPrefix(result.Text, " ") || strings.HasSuffix(result.Text, " ") {
		t.Errorf("text should be trimmed: %q", result.Text)
	}
	if !strings.Contains(result.Text, "Multiple") {
		t.Errorf("expected text content, got %q", result.Text)
	}
}

func TestParseHTML_EmptyBody(t *testing.T) {
	result, err := ParseHTML([]byte("<html><body></body></html>"), "https://example.com/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Text != "" {
		t.Errorf("expected empty text for empty body, got %q", result.Text)
	}
}

func TestParseHTML_FallbackOnMalformedHTML(t *testing.T) {
	// Severely malformed HTML — should not panic, should extract whatever text is there
	html := []byte(`<div>Some text here<p>and more text`)
	result, err := ParseHTML(html, "https://example.com/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Text, "Some text here") {
		t.Errorf("expected fallback text extraction, got %q", result.Text)
	}
}

func TestExtractNaive_ExcludesTitleFromBody(t *testing.T) {
	// Unit test on extractNaive directly — the public ParseHTML path rarely
	// reaches this fallback (readability/content-area usually win), but when
	// it does fire the title text must not leak into the body.
	const titleMarker = "UniqueTitleMarker12345"
	htmlBytes := []byte(`<html><head><title>` + titleMarker + `</title></head><body><div>body text here</div></body></html>`)
	doc, err := html.Parse(bytes.NewReader(htmlBytes))
	if err != nil {
		t.Fatalf("html.Parse: %v", err)
	}

	title, text := extractNaive(doc)

	if title != titleMarker {
		t.Errorf("expected title %q, got %q", titleMarker, title)
	}
	if strings.Contains(text, titleMarker) {
		t.Errorf("title marker leaked into body: %q", text)
	}
	if !strings.Contains(text, "body text here") {
		t.Errorf("expected body text to be preserved, got %q", text)
	}
}

func TestNormalizeURL(t *testing.T) {
	stripList := []string{"utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid", "lang"}

	tests := []struct {
		name  string
		raw   string
		strip []string
		want  string
	}{
		{
			name:  "no query untouched",
			raw:   "https://www.uma.es/page",
			strip: stripList,
			want:  "https://www.uma.es/page",
		},
		{
			name:  "strips utm params",
			raw:   "https://www.uma.es/page?utm_source=twitter&utm_medium=social",
			strip: stripList,
			want:  "https://www.uma.es/page",
		},
		{
			name:  "strips mixed tracking and keeps semantic",
			raw:   "https://www.uma.es/page?id=42&utm_campaign=spring&fbclid=abc",
			strip: stripList,
			want:  "https://www.uma.es/page?id=42",
		},
		{
			name:  "strips lang param",
			raw:   "https://www.uma.es/page?lang=es",
			strip: stripList,
			want:  "https://www.uma.es/page",
		},
		{
			name:  "canonicalises query-key order",
			raw:   "https://www.uma.es/page?b=2&a=1",
			strip: stripList,
			want:  "https://www.uma.es/page?a=1&b=2",
		},
		{
			name:  "case-insensitive param name match",
			raw:   "https://www.uma.es/page?UTM_SOURCE=x&id=7",
			strip: stripList,
			want:  "https://www.uma.es/page?id=7",
		},
		{
			name:  "nil strip list leaves query alone",
			raw:   "https://www.uma.es/page?utm_source=x&id=7",
			strip: nil,
			want:  "https://www.uma.es/page?id=7&utm_source=x",
		},
		{
			name:  "removes trailing question mark when all params stripped",
			raw:   "https://www.uma.es/page?utm_source=x",
			strip: stripList,
			want:  "https://www.uma.es/page",
		},
		{
			name:  "invalid url returned as-is",
			raw:   "://bad",
			strip: stripList,
			want:  "://bad",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeURL(tt.raw, tt.strip)
			if got != tt.want {
				t.Errorf("normalizeURL(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestResolveLink(t *testing.T) {
	tests := []struct {
		name string
		href string
		want string
	}{
		{"absolute", "https://www.uma.es/page", "https://www.uma.es/page"},
		{"relative", "page.html", "https://www.uma.es/base/page.html"},
		{"root relative", "/other/page", "https://www.uma.es/other/page"},
		{"fragment only", "#section", ""},
		{"mailto", "mailto:test@uma.es", ""},
		{"javascript", "javascript:void(0)", ""},
		{"tel", "tel:+34123456", ""},
		{"empty", "", ""},
		{"ftp", "ftp://files.uma.es/doc", ""},
		{"strips fragment", "/page#section", "https://www.uma.es/page"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := mustParseURL("https://www.uma.es/base/")
			got := resolveLink(base, tt.href)
			if got != tt.want {
				t.Errorf("resolveLink(%q) = %q, want %q", tt.href, got, tt.want)
			}
		})
	}
}

func TestExtractHTMLLang(t *testing.T) {
	tests := []struct {
		name string
		html string
		want string
	}{
		{
			name: "lowercase es",
			html: `<!DOCTYPE html><html lang="es"><body></body></html>`,
			want: "es",
		},
		{
			name: "lowercase en",
			html: `<html lang="en"></html>`,
			want: "en",
		},
		{
			name: "region-qualified en-US",
			html: `<html lang="en-US"></html>`,
			want: "en",
		},
		{
			name: "region-qualified es-ES",
			html: `<html lang="es-ES"></html>`,
			want: "es",
		},
		{
			name: "uppercase normalised",
			html: `<html lang="ES"></html>`,
			want: "es",
		},
		{
			name: "unsupported language returned empty",
			html: `<html lang="de"></html>`,
			want: "",
		},
		{
			name: "missing lang attr",
			html: `<html></html>`,
			want: "",
		},
		{
			name: "no html element (fragment)",
			html: `<p>text</p>`,
			want: "",
		},
		{
			name: "whitespace trimmed",
			html: `<html lang=" en "></html>`,
			want: "en",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := html.Parse(bytes.NewReader([]byte(tt.html)))
			if err != nil {
				t.Fatalf("html.Parse: %v", err)
			}
			if got := extractHTMLLang(doc); got != tt.want {
				t.Errorf("extractHTMLLang() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseHTML_PopulatesLanguageField(t *testing.T) {
	body := []byte(`<!DOCTYPE html><html lang="en-US"><body><p>Some content here.</p></body></html>`)
	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatalf("ParseHTML: %v", err)
	}
	if result.Language != "en" {
		t.Errorf("expected result.Language = %q, got %q", "en", result.Language)
	}
}

func TestExtractOGTitle(t *testing.T) {
	tests := []struct {
		name string
		html string
		want string
	}{
		{
			name: "og title present",
			html: `<html><head><meta property="og:title" content="My Title"></head><body></body></html>`,
			want: "My Title",
		},
		{
			name: "no og title",
			html: `<html><head><title>Just Title</title></head><body></body></html>`,
			want: "",
		},
		{
			name: "og description ignored",
			html: `<html><head><meta property="og:description" content="Desc"></head><body></body></html>`,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := html.Parse(bytes.NewReader([]byte(tt.html)))
			if err != nil {
				t.Fatal(err)
			}
			if got := extractOGTitle(doc); got != tt.want {
				t.Errorf("title = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseHTML_OGTitlePreferred(t *testing.T) {
	body := []byte(`<html><head>
		<title>Page - Universidad de Málaga</title>
		<meta property="og:title" content="Clean Page Title">
	</head><body><p>Some content here.</p></body></html>`)

	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatal(err)
	}

	if result.Title != "Clean Page Title" {
		t.Errorf("expected og:title to override, got %q", result.Title)
	}
}

func TestRenderTableMarkdown(t *testing.T) {
	tests := []struct {
		name string
		html string
		want string
	}{
		{
			name: "th header row",
			html: `<table><tr><th>Name</th><th>Value</th></tr><tr><td>Credits</td><td>6</td></tr></table>`,
			want: "| Name | Value |\n| --- | --- |\n| Credits | 6 |",
		},
		{
			name: "no th, first row becomes header",
			html: `<table><tr><td>Key</td><td>Val</td></tr><tr><td>A</td><td>B</td></tr></table>`,
			want: "| Key | Val |\n| --- | --- |\n| A | B |",
		},
		{
			name: "single row only",
			html: `<table><tr><td>Only</td><td>Row</td></tr></table>`,
			want: "| Only | Row |\n| --- | --- |",
		},
		{
			name: "thead and tbody",
			html: `<table><thead><tr><th>H1</th><th>H2</th></tr></thead><tbody><tr><td>D1</td><td>D2</td></tr></tbody></table>`,
			want: "| H1 | H2 |\n| --- | --- |\n| D1 | D2 |",
		},
		{
			name: "whitespace in cells",
			html: `<table><tr><td>  Spaced  Out  </td><td>Normal</td></tr></table>`,
			want: "| Spaced Out | Normal |\n| --- | --- |",
		},
		{
			name: "empty table",
			html: `<table></table>`,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := html.Parse(strings.NewReader(tt.html))
			if err != nil {
				t.Fatal(err)
			}
			// Find the <table> node
			var tableNode *html.Node
			var find func(*html.Node)
			find = func(n *html.Node) {
				if n.Type == html.ElementNode && n.Data == "table" {
					tableNode = n
					return
				}
				for c := n.FirstChild; c != nil; c = c.NextSibling {
					find(c)
				}
			}
			find(doc)
			if tableNode == nil {
				t.Fatal("no table node found")
			}

			got := renderTableMarkdown(tableNode)
			if got != tt.want {
				t.Errorf("got:\n%s\nwant:\n%s", got, tt.want)
			}
		})
	}
}

func TestParseHTML_TableInContent(t *testing.T) {
	body := []byte(`<html><head><title>Subject</title></head><body>
		<main>
			<h1>Subject Info</h1>
			<table>
				<tr><th>Field</th><th>Value</th></tr>
				<tr><td>Créditos teóricos</td><td>6</td></tr>
				<tr><td>Créditos prácticos</td><td>3</td></tr>
			</table>
			<p>Additional info here.</p>
		</main>
	</body></html>`)

	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(result.Text, "| Créditos teóricos | 6 |") {
		t.Errorf("expected markdown table in text, got %q", result.Text)
	}
	if !strings.Contains(result.Text, "| Field | Value |") {
		t.Errorf("expected header row in text, got %q", result.Text)
	}
}

func TestParseHTML_MarkdownHeadings(t *testing.T) {
	body := []byte(`<html><head><title>Test</title></head><body>
		<main>
			<h1>Main Title</h1>
			<p>Intro paragraph.</p>
			<h2>Subsection</h2>
			<p>Subsection content.</p>
			<ul>
				<li>Item one</li>
				<li>Item two</li>
			</ul>
		</main>
	</body></html>`)

	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Text, "# Main Title") {
		t.Errorf("expected markdown h1, got %q", result.Text)
	}
	if !strings.Contains(result.Text, "## Subsection") {
		t.Errorf("expected markdown h2, got %q", result.Text)
	}
	if !strings.Contains(result.Text, "- Item one") {
		t.Errorf("expected markdown list items, got %q", result.Text)
	}
}

func TestParseHTML_MarkdownFormatting(t *testing.T) {
	body := []byte(`<html><body>
		<main>
			<p>Text with <strong>bold</strong> and <em>italic</em> words.</p>
			<p>A <a href="https://example.com">link here</a>.</p>
		</main>
	</body></html>`)

	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Text, "**bold**") {
		t.Errorf("expected bold markdown, got %q", result.Text)
	}
	if !strings.Contains(result.Text, "*italic*") {
		t.Errorf("expected italic markdown, got %q", result.Text)
	}
	if !strings.Contains(result.Text, "[link here](https://example.com)") {
		t.Errorf("expected markdown link, got %q", result.Text)
	}
}

func TestNormalizeStructured(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"hello\n\n\n\nworld", "hello\n\nworld"},
		{"  trailing spaces  \n  more  ", "trailing spaces\n  more"},
		{"\n\n\nstart", "start"},
		{"end\n\n\n", "end"},
	}

	for _, tt := range tests {
		got := normalizeStructured(tt.input)
		if got != tt.want {
			t.Errorf("normalizeStructured(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestNormalizeStructuredCollapsesBlankLines(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "three newlines collapsed to two",
			input:    "a\n\n\nb",
			expected: "a\n\nb",
		},
		{
			name:     "many newlines collapsed to two",
			input:    "a\n\n\n\n\n\n\nb",
			expected: "a\n\nb",
		},
		{
			name:     "two newlines preserved",
			input:    "a\n\nb",
			expected: "a\n\nb",
		},
		{
			name:     "trailing spaces trimmed",
			input:    "hello   \nworld  ",
			expected: "hello\nworld",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeStructured(tt.input)
			if got != tt.expected {
				t.Errorf("normalizeStructured(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestNormalizeWhitespace(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"  hello   world  ", "hello world"},
		{"no\nnewlines\there", "no newlines here"},
		{"", ""},
		{"single", "single"},
	}

	for _, tt := range tests {
		got := normalizeWhitespace(tt.input)
		if got != tt.want {
			t.Errorf("normalizeWhitespace(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func makeElement(tag string, attrs ...html.Attribute) *html.Node {
	return &html.Node{
		Type: html.ElementNode,
		Data: tag,
		Attr: attrs,
	}
}

func TestIsBoilerplate(t *testing.T) {
	tests := []struct {
		name string
		node *html.Node
		want bool
	}{
		// Tag-based boilerplate
		{"script tag", makeElement("script"), true},
		{"style tag", makeElement("style"), true},
		{"noscript tag", makeElement("noscript"), true},
		{"nav tag", makeElement("nav"), true},
		{"header tag", makeElement("header"), true},
		{"footer tag", makeElement("footer"), true},
		// Non-boilerplate tags
		{"div tag", makeElement("div"), false},
		{"p tag", makeElement("p"), false},
		{"main tag", makeElement("main"), false},
		{"article tag", makeElement("article"), false},
		// Class-based boilerplate
		{"modal class", makeElement("div", html.Attribute{Key: "class", Val: "modal-dialog"}), true},
		{"cookie class", makeElement("div", html.Attribute{Key: "class", Val: "cookie-banner"}), true},
		{"overlay class", makeElement("div", html.Attribute{Key: "class", Val: "page-overlay"}), true},
		{"beforecontent class", makeElement("div", html.Attribute{Key: "class", Val: "beforecontent-wrapper"}), true},
		{"aftercontent class", makeElement("div", html.Attribute{Key: "class", Val: "aftercontent"}), true},
		{"blockcontainerplace class", makeElement("div", html.Attribute{Key: "class", Val: "blockcontainerplace"}), true},
		{"blockplace class", makeElement("div", html.Attribute{Key: "class", Val: "blockplace-1"}), true},
		// ID-based boilerplate
		{"modal id", makeElement("div", html.Attribute{Key: "id", Val: "modal-login"}), true},
		{"cookie id", makeElement("div", html.Attribute{Key: "id", Val: "cookie-consent"}), true},
		{"beforecontent id", makeElement("div", html.Attribute{Key: "id", Val: "beforecontent01"}), true},
		{"aftercontent id", makeElement("div", html.Attribute{Key: "id", Val: "aftercontent02"}), true},
		// Non-boilerplate with class/id
		{"content class", makeElement("div", html.Attribute{Key: "class", Val: "content-area"}), false},
		{"main id", makeElement("div", html.Attribute{Key: "id", Val: "main_content"}), false},
		// Non-element node
		{"text node", &html.Node{Type: html.TextNode, Data: "hello"}, false},
		// Case insensitivity
		{"MODAL class uppercase", makeElement("div", html.Attribute{Key: "class", Val: "MODAL-Dialog"}), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isBoilerplate(tt.node)
			if got != tt.want {
				t.Errorf("isBoilerplate(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
