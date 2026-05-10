// Package markers defines the inline reference syntax that links a chunk of
// markdown text to image artifacts rendered from the same source document.
//
// The parser emits markers like [[image:p1]] at page boundaries (PDFs) or at
// the top of an HTML page (full-page screenshot). Markers survive chunking
// untouched. The retrieval path scans them to know which artifacts to attach
// to a vision-capable LLM, and strips them from the text shown to the model.
// The embedder also strips them so they don't pollute the embedding space.
//
// The bracket form [[image:ref]] is chosen over <|...|> to avoid colliding
// with tokenizer special tokens (e.g. <|im_start|>, <|endoftext|>).
package markers

import (
	"fmt"
	"regexp"
	"strings"
)

// imageRe matches markers of the form [[image:ref]] where ref is one or more
// alphanumeric or hyphen characters. The capture group yields the ref alone.
var imageRe = regexp.MustCompile(`\[\[image:([a-zA-Z0-9-]+)\]\]`)

// Format returns the marker text for a given ref (e.g. Format("p1") →
// "[[image:p1]]").
func Format(ref string) string {
	return fmt.Sprintf("[[image:%s]]", ref)
}

// Extract returns each ref captured from the input string in source order.
// Duplicates are preserved; callers that want a set should de-dup themselves.
// Returns nil when there are no matches.
func Extract(s string) []string {
	matches := imageRe.FindAllStringSubmatch(s, -1)
	if len(matches) == 0 {
		return nil
	}
	refs := make([]string, len(matches))
	for i, m := range matches {
		refs[i] = m[1]
	}
	return refs
}

// Strip removes every marker from the input. Surrounding whitespace is left
// in place — callers that need cleanup can collapse runs of spaces themselves.
func Strip(s string) string {
	return imageRe.ReplaceAllString(s, "")
}

// Boundary records where a marker appeared in the input, expressed as the byte
// offset in the *cleaned* (marker-free) text. Start is the offset at which the
// marker was located before stripping, so a chunker can intersect cleaned-text
// byte ranges with these boundaries to determine which refs each chunk covers.
type Boundary struct {
	Ref   string
	Start int
}

// ExtractBoundaries strips every marker from input and returns (cleaned, boundaries)
// where each boundary's Start is its byte offset in the cleaned output. When a
// marker is followed by a newline (the parser's "page header" form), the
// newline is also consumed so the boundary points at the first byte of the
// page's content rather than the empty line where the marker was. When the
// input has no markers, returns the original string and a nil slice.
func ExtractBoundaries(s string) (string, []Boundary) {
	locs := imageRe.FindAllStringSubmatchIndex(s, -1)
	if len(locs) == 0 {
		return s, nil
	}
	var b strings.Builder
	b.Grow(len(s))
	boundaries := make([]Boundary, 0, len(locs))
	prev := 0
	for _, loc := range locs {
		b.WriteString(s[prev:loc[0]])
		boundaries = append(boundaries, Boundary{
			Ref:   s[loc[2]:loc[3]],
			Start: b.Len(),
		})
		end := loc[1]
		if end < len(s) && s[end] == '\n' {
			end++
		}
		prev = end
	}
	b.WriteString(s[prev:])
	return b.String(), boundaries
}
