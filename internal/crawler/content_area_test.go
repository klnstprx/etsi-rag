package crawler

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"golang.org/x/net/html"
)

func TestExtractContentArea_SubjectPage(t *testing.T) {
	body, err := os.ReadFile("/tmp/subject.html")
	if err != nil {
		t.Skip("no /tmp/subject.html — run: curl -sL 'https://www.uma.es/centers/subject/5157/53412' > /tmp/subject.html")
	}

	doc, err := html.Parse(bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	title, text := extractContentArea(doc)
	normalized := normalizeWhitespace(text)

	t.Logf("Title: %s", title)
	t.Logf("Content length: %d", len(normalized))
	t.Logf("Content: %s", normalized)

	// Should contain structured data
	for _, want := range []string{
		"Información detallada",
		"Créditos teóricos",
		"Guía Docente",
		"Profesores:",
	} {
		if !strings.Contains(normalized, want) {
			t.Errorf("missing expected content: %q", want)
		}
	}

	// Should NOT contain nav boilerplate
	for _, unwanted := range []string{
		"Conoce la UMA",
		"Botones de cabecera",
		"Defensoría Universitaria",
		"Trozos html editables",
		"Antes del cuerpo del contenido",
	} {
		if strings.Contains(normalized, unwanted) {
			t.Errorf("contains boilerplate: %q", unwanted)
		}
	}
}

func TestParseHTML_SubjectPage(t *testing.T) {
	body, err := os.ReadFile("/tmp/subject.html")
	if err != nil {
		t.Skip("no /tmp/subject.html")
	}

	result, err := ParseHTML(body, "https://www.uma.es/centers/subject/5157/53412")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("ParseHTML text length: %d", len(result.Text))
	t.Logf("ParseHTML text: %s", result.Text)

	// Should contain professor info and subject details
	for _, want := range []string{
		"Créditos teóricos",
		"Guía Docente",
		"Profesores:",
	} {
		if !strings.Contains(result.Text, want) {
			t.Errorf("missing expected content: %q", want)
		}
	}

	// Links should include the guia docente PDF
	var hasSara bool
	for _, l := range result.Links {
		if strings.Contains(l, "sara.uma.es") && strings.Contains(l, "Programas") {
			hasSara = true
			break
		}
	}
	if !hasSara {
		t.Error("missing sara.uma.es guia docente PDF link")
	}
}

func TestParseHTML_CmsPage(t *testing.T) {
	body, err := os.ReadFile("/tmp/etsi_home.html")
	if err != nil {
		t.Skip("no /tmp/etsi_home.html")
	}

	result, err := ParseHTML(body, "https://www.uma.es/etsi-informatica/")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Title: %s", result.Title)
	t.Logf("Text length: %d", len(result.Text))
	t.Logf("Links count: %d", len(result.Links))

	// Should have meaningful content
	if len(result.Text) < 100 {
		t.Errorf("text too short: %d chars", len(result.Text))
	}
	// Should have links
	if len(result.Links) < 10 {
		t.Errorf("too few links: %d", len(result.Links))
	}
}
