package crawler

import (
	"bytes"
	"net/url"
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

	text := extractNaive(doc)

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
			name:  "uma.es http upgraded to https and host lowercased",
			raw:   "http://WWW.uma.es/etsi-informatica/",
			strip: stripList,
			want:  "https://www.uma.es/etsi-informatica/",
		},
		{
			name:  "uma.es subdomain upgraded to https with query",
			raw:   "http://sara.uma.es/ht/x.pdf?b=2&a=1",
			strip: stripList,
			want:  "https://sara.uma.es/ht/x.pdf?a=1&b=2",
		},
		{
			name:  "non-uma host keeps its scheme",
			raw:   "http://Example.com/page",
			strip: stripList,
			want:  "http://example.com/page",
		},
		{
			name:  "lookalike host keeps its scheme",
			raw:   "http://notuma.es/page",
			strip: stripList,
			want:  "http://notuma.es/page",
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

	base, err := url.Parse("https://www.uma.es/base/")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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

func TestParseHTML_PlaceholderOGTitleFallsBackToDocumentTitle(t *testing.T) {
	// UMA's CMS stamps og:title with its internal node name on course pages.
	body := []byte(`<html><head><meta charset="utf-8"><title>Almacenes de Datos - Universidad de Málaga</title>` +
		`<meta property="og:title" content="subject1"></head>` +
		`<body><main><h1>Almacenes de Datos</h1><p>Información detallada de la asignatura con texto suficiente para extraer.</p>` +
		`<svg><title>buscar</title><path d="M0 0"/></svg></main></body></html>`)
	result, err := ParseHTML(body, "https://www.uma.es/centers/subject/5157/53412")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Title == "subject1" || !strings.Contains(result.Title, "Almacenes de Datos") {
		t.Errorf("Title = %q, want the page's real title", result.Title)
	}
}

func TestParseHTML_UMASiteRootDropsHiddenCMSBlocks(t *testing.T) {
	// Trimmed from the live grado-en-ingenieria-informatica/ page: the iDUMA
	// session modal and the CMS editor placeholders sit next to the content.
	result, err := ParseHTML(loadFixture(t, "uma_grado_home.html"), "https://www.uma.es/grado-en-ingenieria-informatica/")
	if err != nil {
		t.Fatal(err)
	}
	for _, junk := range []string{"cierre de sesión", "Trozos html", "beforecontenttitle", "Antes del cuerpo del contenido"} {
		if strings.Contains(result.Text, junk) {
			t.Errorf("text contains hidden CMS block %q:\n%s", junk, result.Text)
		}
	}
	if !strings.Contains(result.Text, "### Datos Generales del Título") {
		t.Errorf("text lost the content heading:\n%s", result.Text)
	}
	if result.Title != "Grado en INGENIERÍA INFORMÁTICA" {
		t.Errorf("Title = %q, want the site name from <title>", result.Title)
	}
}

func TestParseHTML_PagesWithoutContentIgnoreSessionModal(t *testing.T) {
	// Trimmed from master-en-ciberseguridad/cms/menu/plan-de-estudios/: no
	// content container, only the hidden iDUMA modal and an empty-menu notice.
	// The second page swaps the notice for the note of an iframe-only page
	// (the máster horarios pages). Each page keeps its own text, so the shared
	// modal no longer makes different pages hash to one document.
	emptyMenu := loadFixture(t, "uma_empty_menu.html")
	iframePage := bytes.Replace(emptyMenu,
		[]byte(`<p class="emptyMenu">This menu does not link to any target</p>`),
		[]byte(`<p>Horarios del primer semestre.</p><iframe src="https://docs.google.com/spreadsheets/d/x/pubhtml"></iframe>`), 1)

	var texts []string
	for _, page := range []struct {
		body []byte
		own  string
	}{
		{emptyMenu, "This menu does not link to any target"},
		{iframePage, "Horarios del primer semestre."},
	} {
		result, err := ParseHTML(page.body, "https://www.uma.es/master-en-ciberseguridad/cms/menu/plan-de-estudios/")
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(result.Text, "cierre de sesión") || strings.Contains(result.Text, "Sesión terminada") {
			t.Errorf("text is the hidden session modal:\n%s", result.Text)
		}
		if !strings.Contains(result.Text, page.own) {
			t.Errorf("text lost the page's own content %q:\n%s", page.own, result.Text)
		}
		texts = append(texts, result.Text)
	}
	if texts[0] == texts[1] {
		t.Errorf("two different pages without a content container produced the same text:\n%s", texts[0])
	}
}

func TestParseHTML_SubpageKeepsOGTitle(t *testing.T) {
	// Below a site's root, og:title is the page's own title even though
	// <title> also starts with the site name.
	body := []byte("<html><head><title> E.T.S.I. INFORMÁTICA -  Máster Universitario en Ing. Informática | Calendario\n - Universidad de Málaga</title>" +
		`<meta property="og:title" content="Máster Universitario en Ing. Informática | Calendario"></head>` +
		`<body><main><p>Calendario académico del máster con las fechas de inicio y fin de cada semestre.</p></main></body></html>`)
	result, err := ParseHTML(body, "https://www.uma.es/etsi-informatica/info/154166/master-en-ing-informatica-calendario/")
	if err != nil {
		t.Fatal(err)
	}
	if want := "Máster Universitario en Ing. Informática | Calendario"; result.Title != want {
		t.Errorf("Title = %q, want %q", result.Title, want)
	}
}

func TestParseHTML_SiteRootKeepsSpecificOGTitle(t *testing.T) {
	// UMA root pages whose og:title (also the page's h1) is longer than the
	// site name in <title>: it is the specific name, not a portal name.
	tests := []struct {
		url, docTitle, ogTitle string
	}{
		{"https://www.uma.es/doctorado/", "Doctorado - Escuela de Doctorado Universidad de Málaga - Universidad de Málaga", "Escuela de Doctorado Universidad de Málaga"},
		{"https://www.uma.es/etsi-informatica/", "E.T.S.I. INFORMÁTICA - E.T.S. de Ingeniería Informática - Universidad de Málaga", "E.T.S. de Ingeniería Informática"},
		{"https://www.uma.es/becas/", "BECAS - Becas - Universidad de Málaga", "Becas"},
	}
	for _, tt := range tests {
		body := []byte(`<html><head><title>` + tt.docTitle + `</title>` +
			`<meta property="og:title" content="` + tt.ogTitle + `"></head>` +
			`<body><main><h1>` + tt.ogTitle + `</h1><p>Información general, normativa, calendario y contacto.</p></main></body></html>`)
		result, err := ParseHTML(body, tt.url)
		if err != nil {
			t.Fatal(err)
		}
		if result.Title != tt.ogTitle {
			t.Errorf("%s: Title = %q, want %q", tt.url, result.Title, tt.ogTitle)
		}
	}
}

func TestIsPlaceholderTitle(t *testing.T) {
	tests := []struct {
		title string
		want  bool
	}{
		{"subject1", true},
		{"Presentación", true},
		{"PRESENTACIÓN", true},
		{"Presentación de PowerPoint", true},
		{"Microsoft Word - normativa.doc", true},
		{"Presentación y contacto", false},
		{"Presentación del Grado en Ingeniería Informática", false},
		{"Subject guide", false},
		{"Book of abstracts", false},
		{"Doc. 3 Normativa", false},
		{"Doctorado", false},
	}
	for _, tt := range tests {
		if got := isPlaceholderTitle(tt.title); got != tt.want {
			t.Errorf("isPlaceholderTitle(%q) = %v, want %v", tt.title, got, tt.want)
		}
	}
}

func TestParseHTML_KeepsWordBoundaries(t *testing.T) {
	body := []byte(`<html><head><title>Plan</title></head><body><div id="main_content">
		<h2>Grado en<br>Ingeniería Informática</h2>
		<table>
			<tr><th>Código</th><th>Asignatura</th></tr>
			<tr><td>201</td><td><div><a href="/x/201">Matemática Discreta</a><br/>Discrete Mathematics</div></td></tr>
		</table>
		<p>Plazo: <strong>15 de mayo</strong> hasta el final del curso.</p>
		<p>Patrocinado por la <strong>Cátedra OPPLUS </strong>y la <a href="/tutorias">escuela</a>, consulte las tutorías.</p>
		<p><span>El máster se imparte mediante la impartici</span><span style="font-size:11pt">ó</span><span>n de seminarios.</span></p>
	</div></body></html>`)
	result, err := ParseHTML(body, "https://www.uma.es/etsi-informatica/info/1/plan/")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"## Grado en Ingeniería Informática",
		"| 201 | Matemática Discreta Discrete Mathematics |",
		"Plazo: **15 de mayo** hasta",
		"**Cátedra OPPLUS** y la [escuela](/tutorias), consulte",
		"la impartición de seminarios",
	} {
		if !strings.Contains(result.Text, want) {
			t.Errorf("text missing %q:\n%s", want, result.Text)
		}
	}
}
