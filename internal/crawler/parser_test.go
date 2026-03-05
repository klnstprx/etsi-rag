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

func TestExtractOGMeta(t *testing.T) {
	tests := []struct {
		name  string
		html  string
		wantT string
		wantD string
	}{
		{
			name:  "both og tags",
			html:  `<html><head><meta property="og:title" content="My Title"><meta property="og:description" content="My Description"></head><body></body></html>`,
			wantT: "My Title",
			wantD: "My Description",
		},
		{
			name:  "og title only",
			html:  `<html><head><meta property="og:title" content="Only Title"></head><body></body></html>`,
			wantT: "Only Title",
			wantD: "",
		},
		{
			name:  "og description only",
			html:  `<html><head><meta property="og:description" content="Only Desc"></head><body></body></html>`,
			wantT: "",
			wantD: "Only Desc",
		},
		{
			name:  "fallback to meta name description",
			html:  `<html><head><meta name="description" content="Fallback Desc"></head><body></body></html>`,
			wantT: "",
			wantD: "Fallback Desc",
		},
		{
			name:  "og description preferred over meta name",
			html:  `<html><head><meta property="og:description" content="OG Desc"><meta name="description" content="Meta Desc"></head><body></body></html>`,
			wantT: "",
			wantD: "OG Desc",
		},
		{
			name:  "no meta tags",
			html:  `<html><head><title>Just Title</title></head><body></body></html>`,
			wantT: "",
			wantD: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := html.Parse(bytes.NewReader([]byte(tt.html)))
			if err != nil {
				t.Fatal(err)
			}
			gotT, gotD := extractOGMeta(doc)
			if gotT != tt.wantT {
				t.Errorf("title = %q, want %q", gotT, tt.wantT)
			}
			if gotD != tt.wantD {
				t.Errorf("description = %q, want %q", gotD, tt.wantD)
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

func TestParseHTML_Description(t *testing.T) {
	body := []byte(`<html><head>
		<meta property="og:description" content="A useful summary.">
	</head><body><p>Body text.</p></body></html>`)

	result, err := ParseHTML(body, "https://example.com/")
	if err != nil {
		t.Fatal(err)
	}

	if result.Description != "A useful summary." {
		t.Errorf("expected description %q, got %q", "A useful summary.", result.Description)
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
