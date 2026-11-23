package crawler

import (
	"net/url"
	"strings"
)

// preferCanonical reports whether candidate is a more canonical URL than
// existing for the same content: same host, then fewer path segments, then a
// shorter path, then https, then no query string. UMA serves identical content
// at /grado-en-X/ and its /cms/menu/... sub-pages, and the bare URL must win so
// documents.url does not depend on which page reached the embed stage first.
// The https rule moves documents.url off the http URLs that older crawls
// recorded before normalizeURL put uma.es on https.
func preferCanonical(candidate, existing string) bool {
	if candidate == "" {
		return false
	}
	if existing == "" {
		return true
	}
	cu, cerr := url.Parse(candidate)
	eu, eerr := url.Parse(existing)
	if cerr != nil {
		return false
	}
	if eerr != nil {
		return true
	}
	if !strings.EqualFold(cu.Host, eu.Host) {
		return false
	}
	cSegs := pathSegments(cu.Path)
	eSegs := pathSegments(eu.Path)
	if cSegs != eSegs {
		return cSegs < eSegs
	}
	if len(cu.Path) != len(eu.Path) {
		return len(cu.Path) < len(eu.Path)
	}
	if cu.Scheme != eu.Scheme {
		return cu.Scheme == "https"
	}
	return cu.RawQuery == "" && eu.RawQuery != ""
}

// pathSegments counts non-empty path segments. "/" → 0, "/a" → 1, "/a/b/" → 2.
func pathSegments(p string) int {
	n := 0
	for _, seg := range strings.Split(p, "/") {
		if seg != "" {
			n++
		}
	}
	return n
}
