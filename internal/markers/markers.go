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
