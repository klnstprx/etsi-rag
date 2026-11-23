package admin

import (
	"errors"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strconv"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
)

const docsPageSize = 50

// DocumentList handles GET /admin/documents.
//
// Query params: q (search), source (source key), tag (single tag). The tag
// filter resolves to the set of source keys whose Tags slice contains it; an
// explicit `source` filter narrows that set further. When the registry has
// no source matching the tag, the result is forced empty rather than letting
// SQL fall back to "no filter" — otherwise a typo'd tag would silently
// return everything.
func (h *Handler) DocumentList(w http.ResponseWriter, r *http.Request) {
	data, err := h.documentsData(r, docFilterFrom(r.URL.Query()))
	if err != nil {
		h.logger.Error().Err(err).Msg("listing documents")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	render(w, r, templates.DocumentsPage(data), templates.DocumentsContent(data))
}

// docFilter is the documents list view state carried in the page URL.
type docFilter struct {
	search, source, tag string
	page                int
}

// docFilterFrom reads the list view state from URL query values; a missing or
// invalid page means the first page.
func docFilterFrom(q url.Values) docFilter {
	page, _ := strconv.Atoi(q.Get("page"))
	return docFilter{search: q.Get("q"), source: q.Get("source"), tag: q.Get("tag"), page: max(page, 1)}
}

// documentsData resolves the source/tag filter, queries the store, and
// returns the populated DocumentsData. Shared by DocumentList and the
// post-delete re-render in DocumentDelete so they cannot drift on filter
// semantics.
func (h *Handler) documentsData(r *http.Request, f docFilter) (templates.DocumentsData, error) {
	allSources := h.snapshotSources()
	filterKeys, forceEmpty := resolveSourceFilter(allSources, f.source, f.tag)

	var docs []store.DocumentSummary
	var total int
	if !forceEmpty {
		filter := store.DocumentListFilter{
			Search:     f.search,
			SourceKeys: filterKeys,
		}
		var err error
		docs, total, err = h.store.ListDocumentsSummary(r.Context(), docsPageSize, (f.page-1)*docsPageSize, filter)
		if err != nil {
			return templates.DocumentsData{}, err
		}
	}

	return templates.DocumentsData{
		Documents:     docs,
		Total:         total,
		Page:          f.page,
		PageSize:      docsPageSize,
		Search:        f.search,
		Source:        f.source,
		Tag:           f.tag,
		SourceOptions: allSources,
		SourceLabels:  sourceLabelMap(allSources),
		SourceTags:    sourceTagsMap(allSources),
		AllTags:       collectTags(allSources),
	}, nil
}

// DocumentDetail handles GET /admin/documents/{id}.
func (h *Handler) DocumentDetail(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		http.Error(w, "invalid document id", http.StatusBadRequest)
		return
	}

	doc, err := h.store.Document(r.Context(), id)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			http.Error(w, "document not found", http.StatusNotFound)
			return
		}
		h.logger.Error().Err(err).Str("id", id).Msg("getting document")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	chunks, err := h.store.ChunksByDocument(r.Context(), id)
	if err != nil {
		h.logger.Error().Err(err).Str("id", id).Msg("getting chunks")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	sourceKey, sourceName, tags := h.documentSourceMeta(r, id)

	data := templates.DocumentDetailData{
		Document:   *doc,
		Chunks:     chunks,
		SourceKey:  sourceKey,
		SourceName: sourceName,
		Tags:       tags,
	}

	pg := templates.DocumentDetailPage(data)
	content := templates.DocumentDetailContent(data)
	render(w, r, pg, content)
}

// DocumentDelete handles DELETE /admin/documents/{id}.
func (h *Handler) DocumentDelete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		http.Error(w, "invalid document id", http.StatusBadRequest)
		return
	}

	if err := h.store.DeleteDocument(r.Context(), id); err != nil {
		h.logger.Error().Err(err).Str("id", id).Msg("deleting document")
		writeToast(w, r, "Failed to delete document: "+err.Error(), "error")
		return
	}

	// Re-render the list with the user's active filters, which HTMX passes as
	// the browser URL in HX-Current-URL. An absent or unparseable header
	// renders the unfiltered first page.
	var view url.Values
	if u, err := url.Parse(r.Header.Get("HX-Current-URL")); err == nil {
		view = u.Query()
	}
	data, err := h.documentsData(r, docFilterFrom(view))
	if err != nil {
		h.logger.Error().Err(err).Msg("listing documents")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeHTML(w, r, templates.DocumentsContent(data), templates.Toast("Document deleted", "success"))
}

// snapshotSources returns the registry snapshot, or nil when no registry is
// wired (e.g. in unit tests that don't seed sources.toml). Callers must
// tolerate a nil/empty slice — that just means no source/tag filters can be
// applied.
func (h *Handler) snapshotSources() []domain.CrawlSource {
	if h.sources == nil {
		return nil
	}
	return h.sources.Snapshot()
}

// resolveSourceFilter combines the explicit `source` query param and the
// `tag` query param into a single set of source keys for the store filter.
//
// Returns (keys, forceEmpty). `forceEmpty=true` means the user requested a
// filter (a tag, or a tag+source pair) but no sources match — the caller
// should bypass the store query and render an empty table. An empty `keys`
// slice with forceEmpty=false means "no source filter": fetch everything.
func resolveSourceFilter(all []domain.CrawlSource, source, tag string) ([]string, bool) {
	switch {
	case source == "" && tag == "":
		return nil, false
	case source != "" && tag == "":
		// Explicit source key — pass through even if it isn't in the
		// registry (the registry could have been edited after the doc was
		// ingested; the user may still want to see those rows).
		return []string{source}, false
	}

	tagged := domain.SourceKeysWithTag(all, tag)
	if source == "" {
		if len(tagged) == 0 {
			return nil, true
		}
		return tagged, false
	}

	// Both filters set: intersect.
	if slices.Contains(tagged, source) {
		return []string{source}, false
	}
	return nil, true
}

// sourceLabelMap builds a key → display-name lookup for fast row rendering.
func sourceLabelMap(sources []domain.CrawlSource) map[string]string {
	m := make(map[string]string, len(sources))
	for _, s := range sources {
		m[s.Key] = s.Name
	}
	return m
}

// sourceTagsMap builds a key → tags lookup so each row's tag pills can be
// resolved in O(1) at render time.
func sourceTagsMap(sources []domain.CrawlSource) map[string][]string {
	m := make(map[string][]string, len(sources))
	for _, s := range sources {
		m[s.Key] = s.Tags
	}
	return m
}

// collectTags returns the deduplicated, sorted set of tags across every
// source — populates the Tag filter dropdown.
func collectTags(sources []domain.CrawlSource) []string {
	seen := make(map[string]struct{})
	for _, s := range sources {
		for _, t := range s.Tags {
			if t == "" {
				continue
			}
			seen[t] = struct{}{}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	return slices.Sorted(maps.Keys(seen))
}

// documentSourceMeta resolves the source key + display name + tags for a
// document by joining through crawl_pages and the in-memory sources registry.
// All return values can be zero/nil when the document has no crawl_page or
// the source is no longer registered — the detail view simply omits those
// fields in that case.
func (h *Handler) documentSourceMeta(r *http.Request, docID string) (*string, string, []string) {
	key, err := h.store.SourceKeyByDocID(r.Context(), docID)
	if err != nil {
		h.logger.Warn().Err(err).Str("id", docID).Msg("looking up source for document")
		return nil, "", nil
	}
	if key == "" {
		return nil, "", nil
	}

	for _, s := range h.snapshotSources() {
		if s.Key == key {
			return &key, s.Name, s.Tags
		}
	}
	return &key, "", nil
}
