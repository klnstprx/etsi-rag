package crawler

import (
	"context"
	"os/exec"
	"testing"

	"etsi-rag/internal/config"
)

var defaultPDFConfig = config.PDFConfig{
	HeadingPatterns: []string{
		"DESCRIPCION", "OBJETIVOS", "CONTENIDOS", "METODOLOGIA",
		"EVALUACION", "COMPETENCIAS", "TEMARIO", "BIBLIOGRAFIA",
		"ACTIVIDADES FORMATIVAS", "SISTEMA DE EVALUACION",
		"RESULTADOS DE APRENDIZAJE", "EQUIPO DOCENTE",
		"RECOMENDACIONES", "DISTRIBUCION DEL TRABAJO",
	},
}

func TestIsAllCaps(t *testing.T) {
	tests := []struct {
		name string
		line string
		want bool
	}{
		{"simple uppercase", "HELLO WORLD", true},
		{"spanish accented", "DESCRIPCIÓN DE LA ASIGNATURA", true},
		{"with digits", "CURSO 2025 / 2026", true},
		{"with punctuation", "OBJETIVOS Y RESULTADOS:", true},
		{"mixed case", "Hello World", false},
		{"all lowercase", "hello world", false},
		{"empty string", "", false},
		{"only spaces", "   ", false},
		{"single uppercase word", "OBJETIVOS", true},
		{"with parentheses", "CONTENIDOS (PROGRAMA)", true},
		{"lowercase word mixed in", "CONTENIDOS de la ASIGNATURA", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isAllCaps(tt.line)
			if got != tt.want {
				t.Errorf("isAllCaps(%q) = %v, want %v", tt.line, got, tt.want)
			}
		})
	}
}

func TestIsHeadingLine(t *testing.T) {
	patterns := []string{"DESCRIPCION", "OBJETIVOS", "CONTENIDOS", "EVALUACION"}

	tests := []struct {
		name string
		line string
		want bool
	}{
		{"matches pattern", "DESCRIPCION DE LA ASIGNATURA", true},
		{"matches pattern accented", "DESCRIPCIÓN DE LA ASIGNATURA", true},
		{"no pattern no match", "PROGRAMA DETALLADO DEL CURSO", false},
		{"short no pattern", "NOTA", false},
		{"not all caps", "Descripcion de la Asignatura", false},
		{"empty", "", false},
		{"short all caps matching pattern", "OBJETIVOS", true},
		{"all caps no pattern short", "ABC", false},
		{"professor name false positive", "FRANCISCO JAVIER", false},
		{"department name false positive", "ARQUITECTURA DE COMPUTADORES", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isHeadingLine(tt.line, patterns)
			if got != tt.want {
				t.Errorf("isHeadingLine(%q, patterns) = %v, want %v", tt.line, got, tt.want)
			}
		})
	}
}

func TestToTitleCase(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"simple", "HELLO WORLD", "Hello World"},
		{"spanish stop words", "DESCRIPCION DE LA ASIGNATURA", "Descripcion de la Asignatura"},
		{"first word is stop word", "DE LA ASIGNATURA", "De la Asignatura"},
		{"with accents", "DESCRIPCIÓN DE LA ASIGNATURA", "Descripción de la Asignatura"},
		{"multiple stop words", "RESULTADOS DE APRENDIZAJE Y COMPETENCIAS", "Resultados de Aprendizaje y Competencias"},
		{"single word", "OBJETIVOS", "Objetivos"},
		{"empty", "", ""},
		{"with numbers", "CURSO 2025", "Curso 2025"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toTitleCase(tt.input)
			if got != tt.want {
				t.Errorf("toTitleCase(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestStripPageHeaders(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "repeating header across pages",
			text: "GUÍA DOCENTE\nReal content page 1\f" +
				"GUÍA DOCENTE\nReal content page 2\f" +
				"GUÍA DOCENTE\nReal content page 3",
			want: "Real content page 1\n\nReal content page 2\n\nReal content page 3",
		},
		{
			name: "page numbers stripped",
			text: "Content here\nPágina 1 de 4\f" +
				"More content\nPágina 2 de 4",
			want: "Content here\n\nMore content",
		},
		{
			name: "single page no repeating headers",
			text: "GUÍA DOCENTE\nSome content here",
			want: "GUÍA DOCENTE\nSome content here",
		},
		{
			name: "empty text",
			text: "",
			want: "",
		},
		{
			name: "pagina case insensitive",
			text: "Content\npagina 1 de 3\fMore\npagina 2 de 3",
			want: "Content\n\nMore",
		},
		{
			name: "split page number Página X de on its own line",
			text: "Content\nPágina 1 de\f" +
				"More content\nPágina 2 de",
			want: "Content\n\nMore content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripPageHeaders(tt.text)
			if got != tt.want {
				t.Errorf("stripPageHeaders() =\n%q\nwant:\n%q", got, tt.want)
			}
		})
	}
}

func TestPdfTextToMarkdown(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "headings converted to markdown",
			raw: "DESCRIPCION DE LA ASIGNATURA\n\n" +
				"This is a course about programming.\n\n" +
				"OBJETIVOS Y RESULTADOS DE APRENDIZAJE\n\n" +
				"Students will learn to code.",
			want: "## Descripcion de la Asignatura\n\n" +
				"This is a course about programming.\n\n" +
				"## Objetivos y Resultados de Aprendizaje\n\n" +
				"Students will learn to code.",
		},
		{
			name: "repeating headers stripped across pages",
			raw: "GUÍA DOCENTE\nDESCRIPCION DE LA ASIGNATURA\n\nContent page 1\nPágina 1 de 2\f" +
				"GUÍA DOCENTE\nOBJETIVOS\n\nContent page 2\nPágina 2 de 2",
			want: "## Descripcion de la Asignatura\n\nContent page 1\n\n## Objetivos\n\nContent page 2",
		},
		{
			name: "plain text without headings preserved",
			raw:  "Just some regular text.\n\nWith paragraph breaks.",
			want: "Just some regular text.\n\nWith paragraph breaks.",
		},
		{
			name: "professor names not promoted to headings",
			raw: "EQUIPO DOCENTE\n\nCOORDINADOR/A\n\n" +
				"FRANCISCO JAVIER\nHORMIGO AGUILAR\n\n" +
				"ARQUITECTURA DE COMPUTADORES",
			want: "## Equipo Docente\n\nCOORDINADOR/A\n\n" +
				"FRANCISCO JAVIER\nHORMIGO AGUILAR\n\n" +
				"ARQUITECTURA DE COMPUTADORES",
		},
		{
			name: "empty input",
			raw:  "",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pdfTextToMarkdown(tt.raw, defaultPDFConfig)
			if got != tt.want {
				t.Errorf("pdfTextToMarkdown() =\n%q\nwant:\n%q", got, tt.want)
			}
		})
	}
}

func TestExtractPDFText_Integration(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not available")
	}

	// Minimal PDF with known content for integration testing.
	// If a fixture exists in testdata/, use it; otherwise skip.
	t.Run("returns structured text", func(t *testing.T) {
		t.Skip("needs testdata/guia_docente.pdf fixture — run manually with a real PDF")

		// Placeholder for when we have a test fixture:
		// pdfData, err := os.ReadFile("testdata/guia_docente.pdf")
		// if err != nil { t.Fatal(err) }
		// result, err := ExtractPDFText(context.Background(), pdfData, defaultPDFConfig)
		// if err != nil { t.Fatal(err) }
		// if !strings.Contains(result.Text, "## ") { t.Error("expected markdown headings") }
	})
}

func TestExtractPDFTitle_Integration(t *testing.T) {
	if _, err := exec.LookPath("pdfinfo"); err != nil {
		t.Skip("pdfinfo not available")
	}

	// Title extraction is non-fatal — empty string on error is fine.
	ctx := context.Background()
	title := extractPDFTitle(ctx, "/nonexistent/file.pdf")
	if title != "" {
		t.Errorf("expected empty title for nonexistent file, got %q", title)
	}
}
