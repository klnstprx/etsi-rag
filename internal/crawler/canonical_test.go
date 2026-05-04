package crawler

import "testing"

func TestPreferCanonical(t *testing.T) {
	tests := []struct {
		name      string
		candidate string
		existing  string
		want      bool
		why       string
	}{
		// The motivating case: bare program URL beats /cms/menu/... sub-page.
		{
			name:      "bare landing beats /cms/menu sub-page",
			candidate: "https://www.uma.es/grado-en-ingenieria-del-software/",
			existing:  "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/",
			want:      true,
			why:       "1 segment vs 4; bare URL is canonical",
		},
		{
			name:      "/cms/menu sub-page does NOT replace bare landing",
			candidate: "https://www.uma.es/grado-en-ingenieria-del-software/cms/menu/informacion-grado/",
			existing:  "https://www.uma.es/grado-en-ingenieria-del-software/",
			want:      false,
			why:       "deeper path — never displaces a shorter one",
		},

		// Empty handling.
		{name: "empty candidate never wins", candidate: "", existing: "https://x/y", want: false},
		{name: "empty existing always loses", candidate: "https://x/y", existing: "", want: true},

		// Cross-host: never replace.
		{
			name:      "cross-host candidate is rejected",
			candidate: "https://other.uma.es/path",
			existing:  "https://www.uma.es/path/with/depth",
			want:      false,
			why:       "host swap is never canonical, even at shorter depth",
		},

		// Same depth: shortest path string wins.
		{
			name:      "same segment count, candidate is shorter — wins",
			candidate: "https://www.uma.es/a/b",
			existing:  "https://www.uma.es/long/page",
			want:      true,
			why:       "/a/b (4 chars) vs /long/page (10) — same 2 segments, shorter wins",
		},
		{
			name:      "same segment count, candidate is longer — loses",
			candidate: "https://www.uma.es/aaaaa/bbbbb",
			existing:  "https://www.uma.es/a/b",
			want:      false,
			why:       "candidate is longer at same depth — existing keeps its slot",
		},

		// Trailing slash should not flip the result by itself.
		{
			name:      "trailing slash on candidate doesn't make it canonical-er",
			candidate: "https://www.uma.es/a/",
			existing:  "https://www.uma.es/a",
			want:      false,
			why:       "same 1 segment; existing is shorter (no slash) — keep it",
		},

		// Idempotence: same URL on both sides.
		{
			name:      "identical URLs do not swap",
			candidate: "https://www.uma.es/a",
			existing:  "https://www.uma.es/a",
			want:      false,
		},

		// Non-program seed shape: /etsi-informatica/info/.../detail beating /cms/menu inside it.
		{
			name:      "etsi-informatica info prefers shorter path",
			candidate: "https://www.uma.es/etsi-informatica/info/12345/page",
			existing:  "https://www.uma.es/etsi-informatica/info/12345/page/cms/menu/x",
			want:      true,
		},

		// Malformed candidate URL: never replace.
		{
			name:      "malformed candidate never wins",
			candidate: "ht!tp://broken url with spaces",
			existing:  "https://www.uma.es/ok",
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := preferCanonical(tt.candidate, tt.existing)
			if got != tt.want {
				t.Errorf("preferCanonical(%q, %q) = %v, want %v (%s)",
					tt.candidate, tt.existing, got, tt.want, tt.why)
			}
		})
	}
}

func TestPathSegments(t *testing.T) {
	tests := []struct {
		path string
		want int
	}{
		{"", 0},
		{"/", 0},
		{"/a", 1},
		{"/a/", 1},
		{"/a/b", 2},
		{"/a/b/", 2},
		{"/grado-en-X", 1},
		{"/grado-en-X/cms/menu/info", 4},
		{"//double/slash", 2}, // empty segments don't count
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			if got := pathSegments(tt.path); got != tt.want {
				t.Errorf("pathSegments(%q) = %d, want %d", tt.path, got, tt.want)
			}
		})
	}
}
