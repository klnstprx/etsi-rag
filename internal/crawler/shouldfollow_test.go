package crawler

import "testing"

func TestShouldFollow(t *testing.T) {
	seedPath := "/etsi-informatica/info/"

	tests := []struct {
		name     string
		link     string
		seedPath string
		want     bool
	}{
		// Existing behavior
		{
			name:     "same host under seed path",
			link:     "https://www.uma.es/etsi-informatica/info/page",
			seedPath: seedPath,
			want:     true,
		},
		{
			name:     "pdf under media files",
			link:     "https://www.uma.es/media/files/document.pdf",
			seedPath: seedPath,
			want:     true,
		},
		{
			name:     "sara guia docente",
			link:     "https://sara.uma.es/ht/2025/ProgramasAsignaturas_Titulacion_5102.pdf",
			seedPath: seedPath,
			want:     true,
		},
		{
			name:     "skip news",
			link:     "https://www.uma.es/etsi-informatica/info/noticias/latest",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "external host",
			link:     "https://google.com/search",
			seedPath: seedPath,
			want:     false,
		},

		// Binary/non-content extensions that MUST be blocked
		{
			name:     "block dotx",
			link:     "https://www.uma.es/media/files/Plantilla_TFG.dotx",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block docx",
			link:     "https://www.uma.es/media/files/template.docx",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block pptx",
			link:     "https://www.uma.es/media/files/Presentacion.pptx",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block xlsx",
			link:     "https://www.uma.es/media/files/data.xlsx",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block jpg",
			link:     "https://www.uma.es/media/files/photo.jpg",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block jpeg",
			link:     "https://www.uma.es/media/files/photo.jpeg",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block png",
			link:     "https://www.uma.es/media/files/logo.png",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block gif",
			link:     "https://www.uma.es/media/files/anim.gif",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block svg",
			link:     "https://www.uma.es/media/files/icon.svg",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block webp",
			link:     "https://www.uma.es/media/files/photo.webp",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block zip",
			link:     "https://www.uma.es/media/files/archive.zip",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block mp4",
			link:     "https://www.uma.es/media/files/video.mp4",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block case insensitive PNG",
			link:     "https://www.uma.es/media/files/logo.PNG",
			seedPath: seedPath,
			want:     false,
		},
		{
			name:     "block DOCX uppercase",
			link:     "https://www.uma.es/media/files/template.DOCX",
			seedPath: seedPath,
			want:     false,
		},
		// URL-encoded extensions
		{
			name:     "block dotx with url-encoded path",
			link:     "https://www.uma.es/etsi-informatica/navegador_de_ficheros/docs/descargar/TFG/Plantilla%20memoria%20TFG%202025_Word.dotx",
			seedPath: seedPath,
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldFollow(tt.link, tt.seedPath)
			if got != tt.want {
				t.Errorf("shouldFollow(%q) = %v, want %v", tt.link, got, tt.want)
			}
		})
	}
}
