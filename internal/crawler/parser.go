package crawler

import (
	"net/url"
	"strings"

	"golang.org/x/net/html"
)

// HTMLResult contains the extracted content from an HTML page.
type HTMLResult struct {
	Title string
	Text  string
	Links []string // absolute URLs
}

// ParseHTML extracts title, visible text, and links from raw HTML.
func ParseHTML(body []byte, baseURL string) (*HTMLResult, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}

	doc, err := html.Parse(strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}

	result := &HTMLResult{}
	seen := make(map[string]bool)

	var extract func(*html.Node)
	extract = func(n *html.Node) {
		// Skip non-visible elements
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style", "nav", "header", "footer", "noscript":
				return
			case "title":
				if n.FirstChild != nil && n.FirstChild.Type == html.TextNode {
					result.Title = strings.TrimSpace(n.FirstChild.Data)
				}
			case "a":
				for _, attr := range n.Attr {
					if attr.Key == "href" {
						link := resolveLink(base, attr.Val)
						if link != "" && !seen[link] {
							seen[link] = true
							result.Links = append(result.Links, link)
						}
					}
				}
			}
		}

		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				result.Text += text + " "
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			extract(c)
		}
	}

	extract(doc)
	result.Text = normalizeWhitespace(result.Text)
	return result, nil
}

// resolveLink resolves a relative URL against a base URL.
// Returns empty string for invalid or non-HTTP links.
func resolveLink(base *url.URL, href string) string {
	href = strings.TrimSpace(href)

	// Skip non-HTTP schemes and fragments
	if href == "" || strings.HasPrefix(href, "#") ||
		strings.HasPrefix(href, "mailto:") ||
		strings.HasPrefix(href, "javascript:") ||
		strings.HasPrefix(href, "tel:") {
		return ""
	}

	ref, err := url.Parse(href)
	if err != nil {
		return ""
	}

	resolved := base.ResolveReference(ref)
	// Strip fragment
	resolved.Fragment = ""

	if resolved.Scheme != "http" && resolved.Scheme != "https" {
		return ""
	}

	return resolved.String()
}

// normalizeWhitespace collapses runs of whitespace into single spaces
// and trims the result.
func normalizeWhitespace(s string) string {
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}
