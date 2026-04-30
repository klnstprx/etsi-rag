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

func TestExtract(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"none", "no markers here", nil},
		{"single", "intro [[image:p1]] outro", []string{"p1"}},
		{
			"multiple ordered",
			"page1 [[image:p1]] body [[image:p2]] body [[image:p3]]",
			[]string{"p1", "p2", "p3"},
		},
		{
			"with html marker mixed with pdf pages",
			"[[image:full]] hello [[image:p1]]",
			[]string{"full", "p1"},
		},
		{
			"non-marker similar text not matched",
			"[[notimage:p1]] [[image-p1]] [image:p1]",
			nil,
		},
		{
			"duplicates preserved (caller dedupes)",
			"[[image:p1]] x [[image:p1]]",
			[]string{"p1", "p1"},
		},
		{
			"hyphenated ref accepted",
			"[[image:abc-123]]",
			[]string{"abc-123"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Extract(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Extract(%q) = %v, want %v", tt.in, got, tt.want)
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
