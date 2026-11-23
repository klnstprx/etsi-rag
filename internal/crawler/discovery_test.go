package crawler

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/objectstore"
)

// sitePage is one fake response: HTML, a 301 to location, or a bare status.
type sitePage struct {
	html     string
	location string
	status   int
}

// htmlPage is a page whose text names pageURL, so every page hashes to its
// own document, and which links to each of links.
func htmlPage(pageURL string, links ...string) sitePage {
	var b strings.Builder
	fmt.Fprintf(&b, "<html><head><title>%s</title></head><body><main><h1>%s</h1><p>Contenido propio de %s.</p>", pageURL, pageURL, pageURL)
	for _, l := range links {
		fmt.Fprintf(&b, `<a href="%s">enlace</a>`, l)
	}
	b.WriteString("</main></body></html>")
	return sitePage{html: b.String()}
}

// siteTransport sends every request, whatever its host, to the fake site, so
// tests crawl real www.uma.es and sara.uma.es URLs that the scope rules accept.
type siteTransport struct{ addr string }

func (t siteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	out := req.Clone(req.Context())
	out.URL.Scheme, out.URL.Host = "http", t.addr
	out.Host = req.URL.Host
	resp, err := http.DefaultTransport.RoundTrip(out)
	if resp != nil {
		resp.Request = req // the client sees the URL it asked for, as with a real host
	}
	return resp, err
}

// crawlSite runs the crawl pipeline over sources against site, keyed by https
// URL (unknown URLs answer 404). It returns the URLs that got a crawl_pages
// row and how many pages the parse stage took. Sources run one after another
// over one shared frontier, as crawlStageAll does, so the order in which they
// reach a link is fixed.
func crawlSite(t *testing.T, site map[string]sitePage, sources ...domain.CrawlSource) (urls []string, parsed int64) {
	t.Helper()
	ctx := context.Background()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p, ok := site["https://"+r.Host+r.URL.RequestURI()]
		switch {
		case !ok:
			http.NotFound(w, r)
		case p.location != "":
			http.Redirect(w, r, p.location, http.StatusMovedPermanently)
		case p.status != 0:
			w.WriteHeader(p.status)
		default:
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			fmt.Fprint(w, p.html)
		}
	}))
	t.Cleanup(srv.Close)

	bucket := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(bucket.Close)
	obj, err := objectstore.New(config.ObjectStoreConfig{Endpoint: bucket.URL, Bucket: "b", AccessKey: "k", SecretKey: "s", Region: "us-east-1"})
	if err != nil {
		t.Fatalf("objectstore.New: %v", err)
	}

	c, s := newEmbedStageHarness(t)
	c.objStore = obj
	c.config = config.CrawlerConfig{
		MaxDepth:            5,
		MaxPages:            100,
		RequestDelay:        time.Millisecond,
		RequestTimeout:      5 * time.Second,
		ParseWorkers:        1,
		EmbedWorkers:        1,
		MaxEmbedFails:       3,
		ProgressLogInterval: time.Hour,
		StripQueryParams:    []string{"utm_source", "lang", "set_language"},
	}
	c.fetcher = NewFetcher(c.config, zerolog.Nop())
	c.fetcher.client.Transport = siteTransport{addr: srv.Listener.Addr().String()}

	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	r := &run{startedAt: time.Now(), progress: &RunProgress{}, cancel: cancel}
	c.pipeline(runCtx, r, "crawl", false, func(parseCh chan<- CrawlItem) {
		var f frontier
		for _, src := range sources {
			if err := c.crawlSourceStage(runCtx, src, parseCh, r.progress, &f); err != nil {
				t.Errorf("crawling %s: %v", src.Key, err)
			}
		}
	})

	pages, err := s.ListCrawlPages(ctx, nil, nil)
	if err != nil {
		t.Fatalf("ListCrawlPages: %v", err)
	}
	for _, p := range pages {
		urls = append(urls, p.URL)
	}
	slices.Sort(urls)
	return urls, r.progress.Parsed.Load()
}

func source(key, seedURL string) domain.CrawlSource {
	return domain.CrawlSource{Key: key, Name: key, SeedURL: seedURL, Enabled: true}
}

func assertCrawled(t *testing.T, got []string, want ...string) {
	t.Helper()
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("crawl_pages URLs:\n got  %q\n want %q", got, want)
	}
}

// A link that one source finds outside its scope must still be crawled by the
// source whose subtree it belongs to, together with its descendants.
func TestCrawl_LinkOutsideScopeStaysFreeForOwningSource(t *testing.T) {
	const (
		grado = "https://www.uma.es/grado-en-ingenieria-informatica/"
		becas = "https://www.uma.es/becas/"
		sub   = "https://www.uma.es/becas/convocatoria/"
		child = "https://www.uma.es/becas/convocatoria/plazos/"
	)
	site := map[string]sitePage{
		grado: htmlPage(grado, sub), // site-wide menu link, out of the grado's scope
		becas: htmlPage(becas, sub),
		sub:   htmlPage(sub, "plazos/"),
		child: htmlPage(child),
	}

	got, _ := crawlSite(t, site, source("grado", grado), source("becas", becas))

	assertCrawled(t, got, grado, becas, sub, child)
}

// UMA pages link absolute http:// URLs; they must be crawled once, under
// https, and their relative links must stay on https.
func TestCrawl_UMAHttpLinkCrawledOnceUnderHTTPS(t *testing.T) {
	const (
		seed = "https://www.uma.es/etsi-informatica/"
		a    = "https://www.uma.es/etsi-informatica/a/"
		b    = "https://www.uma.es/etsi-informatica/a/b/"
	)
	site := map[string]sitePage{
		seed: htmlPage(seed, "http://www.uma.es/etsi-informatica/a/", a),
		a:    htmlPage(a, "b/"),
		b:    htmlPage(b),
	}

	got, _ := crawlSite(t, site, source("etsi", seed))

	assertCrawled(t, got, seed, a, b)
}

// A redirected page is recorded under its final URL and resolves its links
// against it; a redirect to a URL already queued or outside the source's
// scope is not crawled again.
func TestCrawl_RedirectRecordedUnderFinalURL(t *testing.T) {
	const (
		seed   = "https://www.uma.es/etsi-informatica/"
		target = "https://www.uma.es/etsi-informatica/target/"
		deep   = "https://www.uma.es/etsi-informatica/target/deep/"
		newURL = "https://www.uma.es/etsi-informatica/new/"
		child  = "https://www.uma.es/etsi-informatica/new/child/"
	)
	site := map[string]sitePage{
		seed:                              htmlPage(seed, "moved/", "old/", "away/", "new/"),
		seed + "moved/":                   {location: "/etsi-informatica/target/"},
		target:                            htmlPage(target, "deep/"),
		deep:                              htmlPage(deep),
		seed + "old/":                     {location: newURL},
		newURL:                            htmlPage(newURL, "child/"),
		child:                             htmlPage(child),
		seed + "away/":                    {location: "https://www.uma.es/otro-centro/"},
		"https://www.uma.es/otro-centro/": htmlPage("https://www.uma.es/otro-centro/"),
	}

	got, parsed := crawlSite(t, site, source("etsi", seed))

	assertCrawled(t, got, seed, target, deep, newURL, child)
	if parsed != int64(len(got)) {
		t.Errorf("parse stage took %d pages for %d crawl_pages rows, want one each", parsed, len(got))
	}
}

// Each plan's subject page links its own guía PDF, so the same course under
// two plans is crawled under both; within one plan, the first page kept wins,
// and a URL that fails leaves the plan's other URLs in play.
func TestCrawl_SubjectDedupPerPlan(t *testing.T) {
	const (
		seed      = "https://www.uma.es/grado-en-ingenieria-informatica/"
		failing   = "https://www.uma.es/centers/subject/5296/55153"
		year210   = "https://www.uma.es/centers/subject/etsi-informatica/5296/55153/210"
		year211   = "https://www.uma.es/centers/subject/etsi-informatica/5296/55153/211"
		otherPlan = "https://www.uma.es/centers/subject/5389/55153"
		guia5296  = "https://sara.uma.es/ht/2025/ProgramasAsignaturas_Titulacion_5296_AsigUMA_55153.pdf"
		guia5389  = "https://sara.uma.es/ht/2025/ProgramasAsignaturas_Titulacion_5389_AsigUMA_55153.pdf"
	)
	site := map[string]sitePage{
		seed:      htmlPage(seed, failing, year210, year211, otherPlan),
		failing:   {status: http.StatusInternalServerError},
		year210:   htmlPage(year210, guia5296),
		year211:   htmlPage(year211, guia5296),
		otherPlan: htmlPage(otherPlan, guia5389),
		// The guía PDFs are not served: a failed row still proves they were found.
	}

	got, _ := crawlSite(t, site, source("grado", seed))

	assertCrawled(t, got, seed, failing, year210, otherPlan, guia5296, guia5389)
}

// A subject URL that redirects to a variant already queued yields no page of
// its own, so the variant must still be crawled, guía and all.
func TestCrawl_SubjectRedirectToQueuedVariant(t *testing.T) {
	const (
		seed    = "https://www.uma.es/grado-en-ingenieria-informatica/"
		short   = "https://www.uma.es/centers/subject/5296/55153"
		year210 = "https://www.uma.es/centers/subject/etsi-informatica/5296/55153/210"
		guia    = "https://sara.uma.es/ht/2025/ProgramasAsignaturas_Titulacion_5296_AsigUMA_55153.pdf"
	)
	site := map[string]sitePage{
		seed:    htmlPage(seed, short, year210),
		short:   {location: year210},
		year210: htmlPage(year210, guia),
	}

	got, _ := crawlSite(t, site, source("grado", seed))

	assertCrawled(t, got, seed, year210, guia)
}

// UMA's language switcher (?set_language=) must collapse onto the bare URL
// with the shipped defaults, both from config.default.toml and from the Go
// defaults applied to a config that omits strip_query_params.
func TestNormalizeURL_DefaultsStripLanguageSwitcher(t *testing.T) {
	shipped, err := os.ReadFile("../../config/config.default.toml")
	if err != nil {
		t.Fatalf("reading shipped defaults: %v", err)
	}
	const bare = "https://www.uma.es/etsi-informatica/info/1/buddy-program/"

	for name, toml := range map[string][]byte{"config.default.toml": shipped, "omitted": nil} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			for _, f := range []string{config.DefaultFileName, config.FileName} {
				if err := os.WriteFile(filepath.Join(dir, f), toml, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			cfg, err := config.Load(dir)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if got := normalizeURL(bare+"?set_language=es", cfg.Crawler.StripQueryParams); got != bare {
				t.Errorf("normalizeURL = %q, want %q", got, bare)
			}
		})
	}
}
