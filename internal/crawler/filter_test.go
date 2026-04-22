package crawler

import "testing"

func TestIsLowQualityContent(t *testing.T) {
	// Default rejection patterns shipped in applyDefaults; duplicated here so
	// the filter unit test stays decoupled from config.
	defaultPatterns := []string{
		"página no encontrada",
		"error 404",
		"página en construcción",
		"sitio en construcción",
		"sin descripción disponible",
	}
	// Pattern list with an extra user-supplied pattern, used to exercise
	// substring matching without relying on defaults.
	patternsWithIDUMA := append(append([]string{}, defaultPatterns...), "se ha detectado un cierre de sesión")

	tests := []struct {
		name           string
		text           string
		minContentLen  int
		patterns       []string
		wantLowQuality bool
	}{
		// --- Minimum content length ---
		{
			name:           "empty text",
			text:           "",
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "below threshold",
			text:           "Short text that is not very useful.",
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "at threshold",
			text:           string(make([]byte, 200)),
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},
		{
			name:           "above threshold",
			text:           string(make([]byte, 500)),
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},
		{
			name:           "zero threshold disables length check",
			text:           "x",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},

		// --- User-supplied pattern match (e.g. an admin adding an iDUMA phrase) ---
		{
			name:           "custom pattern — iduma session expired",
			text:           "Se ha detectado un cierre de sesión de su cuenta desde otra aplicación federada con iDUMA o, simplemente, su sesión ha caducado.\n\nPuede autenticarse de nuevo y seguir navegando.",
			minContentLen:  0,
			patterns:       patternsWithIDUMA,
			wantLowQuality: true,
		},
		{
			name:           "custom pattern — iduma mixed case",
			text:           "se ha detectado un cierre de sesión de su cuenta desde otra aplicación",
			minContentLen:  0,
			patterns:       patternsWithIDUMA,
			wantLowQuality: true,
		},
		{
			name:           "iduma phrase NOT in default list → not rejected",
			text:           "se ha detectado un cierre de sesión. " + string(make([]byte, 300)),
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},

		// --- Soft 404 pages ---
		{
			name:           "soft 404 pagina no encontrada",
			text:           "Página no encontrada\n\nLo sentimos, el contenido que buscas no está disponible.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "soft 404 error 404",
			text:           "ERROR 404 — recurso no disponible en este servidor.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "soft 404 uppercase",
			text:           "PÁGINA NO ENCONTRADA. Vuelva al inicio.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},

		// --- Under construction stubs ---
		{
			name:           "under construction pagina",
			text:           "Esta página en construcción. Disculpe las molestias.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "under construction sitio",
			text:           "Sitio en construcción — vuelva pronto.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},

		// --- CMS placeholder ---
		{
			name:           "sin descripcion disponible placeholder",
			text:           "Asignatura: Sistemas Operativos\n\nSin descripción disponible.",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},

		// --- Empty filter pages ---
		{
			name:           "filter stub short",
			text:           "Filtrar por curso:\n\nFiltrar por semestre:\n\nFiltrar por carácter:",
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: true,
		},
		{
			name:           "filter stub with real content after is fine",
			text:           "Filtrar por curso:\n\nFiltrar por semestre:\n\n" + string(make([]byte, 300)),
			minContentLen:  0,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},

		// --- Good content should pass ---
		{
			name:           "guia docente PDF",
			text:           "GUÍA DOCENTE DE LA ASIGNATURA Curso 2025 / 2026 Página 1 de 4 DESCRIPCIÓN DE LA ASIGNATURA Grado/Máster en: Graduado/a en Ingeniería Informática por la Universidad de Málaga. Plan 2010 Centro: Escuela Técnica Superior de Ingeniería Informática Asignatura: Sistemas Operativos",
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},
		{
			name:           "substantial degree page",
			text:           "Información sobre el Grado en Ingeniería Informática de la Universidad de Málaga. El plan de estudios incluye materias de formación básica, obligatorias y optativas distribuidas en cuatro cursos académicos.",
			minContentLen:  200,
			patterns:       defaultPatterns,
			wantLowQuality: false,
		},

		// --- Pattern list customization ---
		{
			name:           "nil patterns skips pattern check",
			text:           "Se ha detectado un cierre de sesión — pero la lista de patrones está vacía.",
			minContentLen:  0,
			patterns:       nil,
			wantLowQuality: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLowQualityContent(tt.text, tt.minContentLen, tt.patterns)
			if got != tt.wantLowQuality {
				t.Errorf("isLowQualityContent(%q..., %d) = %v, want %v",
					truncate(tt.text, 60), tt.minContentLen, got, tt.wantLowQuality)
			}
		})
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func TestSubjectCode(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		// Standard /centers/subject/{plan}/{code} pattern
		{
			name: "two-segment subject URL",
			url:  "https://www.uma.es/centers/subject/5393/55154",
			want: "55154",
		},
		{
			name: "different plan same code",
			url:  "https://www.uma.es/centers/subject/5296/55154",
			want: "55154",
		},
		{
			name: "trailing slash",
			url:  "https://www.uma.es/centers/subject/5381/55772/",
			want: "55772",
		},

		// Extended /centers/subject/{center}/{plan}/{code}/{year?} pattern
		{
			name: "three-segment with center",
			url:  "https://www.uma.es/centers/subject/etsi-informatica/5310/51003/212",
			want: "51003",
		},
		{
			name: "three-segment trailing slash",
			url:  "https://www.uma.es/centers/subject/etsi-informatica/5102/51003/210",
			want: "51003",
		},

		// Non-subject URLs return empty
		{
			name: "not a subject URL",
			url:  "https://www.uma.es/etsi-informatica/info/plan-estudios",
			want: "",
		},
		{
			name: "partial match no code",
			url:  "https://www.uma.es/centers/subject/",
			want: "",
		},
		{
			name: "invalid URL",
			url:  "://bad",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := subjectCode(tt.url)
			if got != tt.want {
				t.Errorf("subjectCode(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}
