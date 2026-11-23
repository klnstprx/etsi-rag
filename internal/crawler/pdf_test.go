package crawler

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/markers"
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
		want []string
	}{
		{
			name: "repeating header across pages",
			text: "GUÍA DOCENTE\nReal content page 1\f" +
				"GUÍA DOCENTE\nReal content page 2\f" +
				"GUÍA DOCENTE\nReal content page 3",
			want: []string{"Real content page 1", "Real content page 2", "Real content page 3"},
		},
		{
			name: "page numbers stripped",
			text: "Content here\nPágina 1 de 4\f" +
				"More content\nPágina 2 de 4",
			want: []string{"Content here", "More content"},
		},
		{
			name: "single page no repeating headers",
			text: "GUÍA DOCENTE\nSome content here",
			want: []string{"GUÍA DOCENTE\nSome content here"},
		},
		{
			name: "empty text",
			text: "",
			want: nil,
		},
		{
			name: "pagina case insensitive",
			text: "Content\npagina 1 de 3\fMore\npagina 2 de 3",
			want: []string{"Content", "More"},
		},
		{
			name: "split page number Página X de on its own line",
			text: "Content\nPágina 1 de\f" +
				"More content\nPágina 2 de",
			want: []string{"Content", "More content"},
		},
		{
			name: "two pages keep their repeated lines",
			text: "Calculo\n15 de junio\fCalculo\n3 de julio",
			want: []string{"Calculo\n15 de junio", "Calculo\n3 de julio"},
		},
		{
			name: "repeated line below the page's first own line stays",
			text: "ACTA\n15 de junio\nCalculo\nAula 1\f" +
				"ACTA\n3 de julio\nCalculo\nAula 2\f" +
				"ACTA\n9 de julio\nCalculo\nAula 3",
			want: []string{"15 de junio\nCalculo\nAula 1", "3 de julio\nCalculo\nAula 2", "9 de julio\nCalculo\nAula 3"},
		},
		{
			name: "repeated line on fewer than 60% of pages stays",
			text: "ACTA\nuno\fACTA\ndos\fACTA\ntres\fcuatro\fcinco\fseis",
			want: []string{"ACTA\nuno", "ACTA\ndos", "ACTA\ntres", "cuatro", "cinco", "seis"},
		},
		{
			name: "blank page keeps its slot",
			text: "Content page 1\f\fContent page 3\f",
			want: []string{"Content page 1", "", "Content page 3"},
		},
		{
			// pdftotext reads a BOE page's footer before its last body line.
			name: "footer read above the page's last body line is stripped",
			text: "BOE\n" + bodyLines(1) + "\ncve: BOE-A-1\nVerificable en boe.es\nlast 1\f" +
				"BOE\n" + bodyLines(2) + "\ncve: BOE-A-1\nVerificable en boe.es\nlast 2\f" +
				"BOE\n" + bodyLines(3) + "\ncve: BOE-A-1\nVerificable en boe.es\nlast 3",
			want: []string{bodyLines(1) + "\nlast 1", bodyLines(2) + "\nlast 2", bodyLines(3) + "\nlast 3"},
		},
		{
			name: "cells repeated within a page stay at its edge",
			text: bodyLines(1) + "\n" + creditRows("Cálculo", "Física", "Lógica") + "\f" +
				bodyLines(2) + "\n" + creditRows("Redes", "Bases de Datos", "Sistemas") + "\f" +
				bodyLines(3) + "\n" + creditRows("Ética", "Gestión", "Compiladores"),
			want: []string{
				bodyLines(1) + "\n" + creditRows("Cálculo", "Física", "Lógica"),
				bodyLines(2) + "\n" + creditRows("Redes", "Bases de Datos", "Sistemas"),
				bodyLines(3) + "\n" + creditRows("Ética", "Gestión", "Compiladores"),
			},
		},
		{
			name: "page total split from Página X de goes, the same number in the body stays",
			text: "GUÍA DOCENTE\nPágina 1 de\n3\nCurso:\n3\f" +
				"GUÍA DOCENTE\nPágina 2 de\n3\nContenido\f" +
				"GUÍA DOCENTE\nPágina 3 de\n3\nBibliografía",
			want: []string{"Curso:\n3", "Contenido", "Bibliografía"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripPageHeaders(tt.text)
			if !slicesEqual(got, tt.want) {
				t.Errorf("stripPageHeaders() =\n%#v\nwant:\n%#v", got, tt.want)
			}
		})
	}
}

// bodyLines returns lines unique to one page, enough to fill both edge zones.
func bodyLines(page int) string {
	lines := make([]string, 2*headerZoneLines)
	for i := range lines {
		lines[i] = fmt.Sprintf("page %d line %d", page, i+1)
	}
	return strings.Join(lines, "\n")
}

// creditRows is a credits table as pdftotext reads it: each subject's name
// followed by the same cells.
func creditRows(subjects ...string) string {
	var rows []string
	for _, s := range subjects {
		rows = append(rows, s, "6", "Créditos ECTS", "100%")
	}
	return strings.Join(rows, "\n")
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPdfTextToMarkdown(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "single page emits one p1 marker at top",
			raw: "DESCRIPCION DE LA ASIGNATURA\n\n" +
				"This is a course about programming.\n\n" +
				"OBJETIVOS Y RESULTADOS DE APRENDIZAJE\n\n" +
				"Students will learn to code.",
			want: "[[image:p1]]\n## Descripcion de la Asignatura\n\n" +
				"This is a course about programming.\n\n" +
				"## Objetivos y Resultados de Aprendizaje\n\n" +
				"Students will learn to code.",
		},
		{
			name: "multi-page emits one marker per page top",
			raw: "GUÍA DOCENTE\nDESCRIPCION DE LA ASIGNATURA\n\nContent page 1\nPágina 1 de 3\f" +
				"GUÍA DOCENTE\nOBJETIVOS\n\nContent page 2\nPágina 2 de 3\f" +
				"GUÍA DOCENTE\nEVALUACION\n\nContent page 3\nPágina 3 de 3\f",
			want: "[[image:p1]]\n## Descripcion de la Asignatura\n\nContent page 1\n\n" +
				"[[image:p2]]\n## Objetivos\n\nContent page 2\n\n" +
				"[[image:p3]]\n## Evaluacion\n\nContent page 3",
		},
		{
			name: "markers keep physical page numbers across a blank page",
			raw:  "Content page 1\f\fContent page 3\f",
			want: "[[image:p1]]\nContent page 1\n\n[[image:p3]]\nContent page 3",
		},
		{
			name: "plain text without headings still gets one marker at top",
			raw:  "Just some regular text.\n\nWith paragraph breaks.",
			want: "[[image:p1]]\nJust some regular text.\n\nWith paragraph breaks.",
		},
		{
			name: "professor names not promoted to headings stay as content",
			raw: "EQUIPO DOCENTE\n\nCOORDINADOR/A\n\n" +
				"FRANCISCO JAVIER\nHORMIGO AGUILAR\n\n" +
				"ARQUITECTURA DE COMPUTADORES",
			want: "[[image:p1]]\n## Equipo Docente\n\nCOORDINADOR/A\n\n" +
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

// skipWithoutPoppler skips a test that runs the poppler CLIs.
func skipWithoutPoppler(t *testing.T) {
	t.Helper()
	for _, bin := range []string{"pdftotext", "pdfinfo", "pdftoppm"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not available", bin)
		}
	}
}

// pdfItem is a crawled PDF as the crawl stage hands it to the parse stage.
func pdfItem(url string, body []byte) CrawlItem {
	return CrawlItem{URL: url, ContentType: "pdf", Body: body}
}

var pdfParseConfig = config.CrawlerConfig{
	PDF: config.PDFConfig{HeadingPatterns: defaultPDFConfig.HeadingPatterns, RenderDPI: 10},
}

// TestParseStage_SubjectGuide runs a real sara.uma.es guide (3 pages, no
// Title metadata) through the parse stage.
func TestParseStage_SubjectGuide(t *testing.T) {
	skipWithoutPoppler(t)
	body, err := os.ReadFile("testdata/guia_docente.pdf")
	if err != nil {
		t.Fatal(err)
	}
	const url = "https://sara.uma.es/ht/2025/ProgramasAsignaturas_Titulacion_5103_AsigUMA_51175.pdf"

	got := runParseStage(t, pdfParseConfig, pdfItem(url, body))
	if len(got) != 1 {
		t.Fatalf("parse stage emitted %d items, want 1", len(got))
	}
	item := got[0]
	wantTitle := "Guía docente de Tecnologías de Aplicaciones Web (Graduado/a en Ingeniería del Software por la Universidad de Málaga. Plan 2010)"
	if item.Title != wantTitle {
		t.Errorf("Title = %q, want %q", item.Title, wantTitle)
	}
	for _, header := range []string{"GUÍA DOCENTE DE LA", "Curso 2025 / 2026", "Página 2 de"} {
		if strings.Contains(item.Text, header) {
			t.Errorf("text keeps running header line %q", header)
		}
	}
	for _, ref := range []string{"p1", "p2", "p3"} {
		if !strings.Contains(item.Text, markers.Format(ref)) {
			t.Errorf("text has no %s marker", ref)
		}
	}
}

func TestParseStage_PDFTitle(t *testing.T) {
	skipWithoutPoppler(t)
	const url = "https://www.uma.es/etsi-informatica/navegador_de_ficheros/docs/descargar/plazas-vacantes.pdf"
	tests := []struct {
		name, title, want string
	}{
		{"single letter", "D", "plazas-vacantes"},
		{"form label", "Destinatario:", "plazas-vacantes"},
		{"template line cut short", "ESCUELA TÉCNICA SUPERIOR DE", "plazas-vacantes"},
		{"authoring filename", "Listado definitivo en inglés.xlsx", "plazas-vacantes"},
		{"first paragraph cut mid-word", "Adjunto le remito, para su conocimiento y efectos oportunos, la composición de los Comités de Mejora para cada una de las titu", "plazas-vacantes"},
		{"real title", "Disposición 7591 del BOE núm. 93 de 2024", "Disposición 7591 del BOE núm. 93 de 2024"},
		{"course number", "Cálculo II", "Cálculo II"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := textPDF("/Title ("+tt.title+")", "Relación de plazas vacantes")
			got := runParseStage(t, pdfParseConfig, pdfItem(url, body))
			if len(got) != 1 {
				t.Fatalf("parse stage emitted %d items, want 1", len(got))
			}
			if got[0].Title != tt.want {
				t.Errorf("Title = %q, want %q", got[0].Title, tt.want)
			}
		})
	}
}

// TestParseStage_PDFPages checks that a table cell repeated on two pages
// survives header stripping and that page markers and renders keep physical
// page numbers across a blank page.
func TestParseStage_PDFPages(t *testing.T) {
	skipWithoutPoppler(t)
	body := textPDF("",
		"Convocatoria ordinaria\nCalculo\n15 de junio",
		"",
		"Convocatoria extraordinaria\nCalculo\n3 de julio",
	)

	got := runParseStage(t, pdfParseConfig, pdfItem("https://www.uma.es/media/files/examenes.pdf", body))
	if len(got) != 1 {
		t.Fatalf("parse stage emitted %d items, want 1", len(got))
	}
	text := got[0].Text
	_, page3, ok := strings.Cut(text, markers.Format("p3"))
	if !ok {
		t.Fatalf("text has no p3 marker:\n%s", text)
	}
	if !strings.Contains(page3, "Calculo") || !strings.Contains(page3, "3 de julio") {
		t.Errorf("page 3 text = %q, want its Calculo row", page3)
	}
	if strings.Count(text, "Calculo") != 2 {
		t.Errorf("text has %d Calculo cells, want 2:\n%s", strings.Count(text, "Calculo"), text)
	}
	if strings.Contains(text, markers.Format("p2")) {
		t.Errorf("blank page 2 got a marker:\n%s", text)
	}
	var refs []string
	for _, a := range got[0].Artifacts {
		refs = append(refs, a.Ref)
	}
	if !slicesEqual(refs, []string{"p1", "p2", "p3"}) {
		t.Errorf("artifact refs = %v, want [p1 p2 p3]", refs)
	}
}

func TestIsPlaceholderPDFTitle(t *testing.T) {
	tests := []struct {
		title string
		want  bool
	}{
		{"subject1", true},
		{"Subject 1", true},
		{"document", true},
		{"Document1", true},
		{"untitled", true},
		{"Sin título", true},
		{"PowerPoint Presentation", true},
		{"Microsoft Word - guia.docx", true},
		{"Slide 1", true},
		{"", true},
		{"Guía Docente Cálculo I", false},
		{"Plan de Estudios GIST", false},
		{"Normativa de permanencia", false},
	}
	for _, tt := range tests {
		if got := isPlaceholderTitle(tt.title); got != tt.want {
			t.Errorf("isPlaceholderTitle(%q) = %v, want %v", tt.title, got, tt.want)
		}
	}
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

// textPDF builds a PDF whose Info dictionary is info (e.g. "/Creator (X)")
// with one page per entry of pages, each line drawn below the previous one.
// Text is Latin-1 (WinAnsiEncoding), so Spanish letters are allowed.
func textPDF(info string, pages ...string) []byte {
	objs := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"", // the page tree, once the pages have object numbers
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
		"<< " + latin1(info) + " >>",
	}
	var kids []string
	for _, page := range pages {
		pageObj := len(objs) + 1
		kids = append(kids, fmt.Sprintf("%d 0 R", pageObj))
		var content strings.Builder
		for i, line := range strings.Split(page, "\n") {
			escaped := strings.NewReplacer(`\`, `\\`, "(", `\(`, ")", `\)`).Replace(latin1(line))
			fmt.Fprintf(&content, "BT /F1 10 Tf 40 %d Td (%s) Tj ET\n", 780-14*i, escaped)
		}
		objs = append(objs,
			fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>", pageObj+1),
			fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", content.Len(), content.String()),
		)
	}
	objs[1] = fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(kids))

	var b strings.Builder
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objs))
	for i, o := range objs {
		offsets[i] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for _, off := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R /Info 4 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, xref)
	return []byte(b.String())
}

// latin1 re-encodes s, which must hold only Latin-1 runes, one byte per rune.
func latin1(s string) string {
	b := make([]byte, 0, len(s))
	for _, r := range s {
		b = append(b, byte(r))
	}
	return string(b)
}

func TestExtractPDFText_TitleOnlyFromTitleMetadata(t *testing.T) {
	if _, err := exec.LookPath("pdfinfo"); err != nil {
		t.Skip("pdfinfo not available")
	}
	tests := []struct {
		name, info, want string
	}{
		{"no Title entry", "/Creator (JasperReports \\(Guia_Programa\\)) /Producer (iText)", ""},
		{"real Title", "/Title (Guia docente) /Creator (JasperReports)", "Guia docente"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractPDFText(context.Background(), textPDF(tt.info, ""), defaultPDFConfig)
			if err != nil {
				t.Fatalf("ExtractPDFText: %v", err)
			}
			if got.Title != tt.want {
				t.Errorf("Title = %q, want %q", got.Title, tt.want)
			}
		})
	}
}
