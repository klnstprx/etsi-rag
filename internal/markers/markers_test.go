package markers

import (
	"reflect"
	"testing"
)

func TestFormat(t *testing.T) {
	tests := []struct {
		ref  string
		want string
	}{
		{"p1", "[[image:p1]]"},
		{"p123", "[[image:p123]]"},
		{"full", "[[image:full]]"},
	}
	for _, tt := range tests {
		t.Run(tt.ref, func(t *testing.T) {
			if got := Format(tt.ref); got != tt.want {
				t.Errorf("Format(%q) = %q, want %q", tt.ref, got, tt.want)
			}
		})
	}
}

func TestExtractBoundaries(t *testing.T) {
	tests := []struct {
		name         string
		in           string
		wantCleaned  string
		wantBoundary []Boundary
	}{
		{"empty", "", "", nil},
		{"no markers", "plain text", "plain text", nil},
		{
			"single marker at start consumes trailing newline",
			"[[image:p1]]\nbody",
			"body",
			[]Boundary{{Ref: "p1", Start: 0}},
		},
		{
			"two pages, marker on its own line",
			"[[image:p1]]\npage one\n\n[[image:p2]]\npage two",
			"page one\n\npage two",
			[]Boundary{{Ref: "p1", Start: 0}, {Ref: "p2", Start: 10}},
		},
		{
			"inline marker mid-text",
			"head [[image:p1]] tail",
			"head  tail",
			[]Boundary{{Ref: "p1", Start: 5}},
		},
		{
			"duplicate refs both reported",
			"[[image:p1]] x [[image:p1]] y",
			" x  y",
			[]Boundary{{Ref: "p1", Start: 0}, {Ref: "p1", Start: 3}},
		},
		{
			"non-marker text untouched",
			"[[notimage:p1]] real [[image:p2]]",
			"[[notimage:p1]] real ",
			[]Boundary{{Ref: "p2", Start: 21}},
		},
		{
			"near-miss syntax not matched",
			"[[image-p1]] [image:p1]",
			"[[image-p1]] [image:p1]",
			nil,
		},
		{
			"hyphenated ref accepted",
			"[[image:abc-123]]",
			"",
			[]Boundary{{Ref: "abc-123", Start: 0}},
		},
		{
			"html and pdf refs in source order",
			"[[image:full]] hello [[image:p1]]",
			" hello ",
			[]Boundary{{Ref: "full", Start: 0}, {Ref: "p1", Start: 7}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cleaned, b := ExtractBoundaries(tt.in)
			if cleaned != tt.wantCleaned {
				t.Errorf("cleaned = %q, want %q", cleaned, tt.wantCleaned)
			}
			if !reflect.DeepEqual(b, tt.wantBoundary) {
				t.Errorf("boundaries = %v, want %v", b, tt.wantBoundary)
			}
		})
	}
}

func TestStrip(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"no markers", "plain text", "plain text"},
		{"removes single", "before [[image:p1]] after", "before  after"},
		{
			"removes multiple",
			"a [[image:p1]] b [[image:p2]] c",
			"a  b  c",
		},
		{
			"leaves non-markers alone",
			"[[notimage:p1]] [[image-p1]] [image:p1]",
			"[[notimage:p1]] [[image-p1]] [image:p1]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Strip(tt.in); got != tt.want {
				t.Errorf("Strip(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
