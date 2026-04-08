package admin

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"

	"etsi-rag/internal/admin/templates"
	"etsi-rag/internal/store"
)

const docsPageSize = 50

// DocumentList handles GET /admin/documents.
func (h *Handler) DocumentList(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}
	search := r.URL.Query().Get("q")
	offset := (page - 1) * docsPageSize

	docs, total, err := h.store.ListDocumentsSummary(r.Context(), docsPageSize, offset, search)
	if err != nil {
		h.logger.Error().Err(err).Msg("listing documents")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	data := templates.DocumentsData{
		Documents: docs,
		Total:     total,
		Page:      page,
		PageSize:  docsPageSize,
		Search:    search,
	}

	pg := templates.DocumentsPage(data)
	content := templates.DocumentsContent(data)
	render(w, r, pg, content)
}

// DocumentDetail handles GET /admin/documents/{id}.
func (h *Handler) DocumentDetail(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		http.Error(w, "invalid document id", http.StatusBadRequest)
		return
	}

	doc, err := h.store.GetDocument(r.Context(), id)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			http.Error(w, "document not found", http.StatusNotFound)
			return
		}
		h.logger.Error().Err(err).Str("id", id).Msg("getting document")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	chunks, err := h.store.GetChunksByDocument(r.Context(), id)
	if err != nil {
		h.logger.Error().Err(err).Str("id", id).Msg("getting chunks")
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	data := templates.DocumentDetailData{
		Document: *doc,
		Chunks:   chunks,
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
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		templates.Toast("Failed to delete document: "+err.Error(), "error").Render(r.Context(), w)
		return
	}

	// Redirect back to list
	docs, total, _ := h.store.ListDocumentsSummary(r.Context(), docsPageSize, 0, "")
	data := templates.DocumentsData{
		Documents: docs,
		Total:     total,
		Page:      1,
		PageSize:  docsPageSize,
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	templates.DocumentsContent(data).Render(r.Context(), w)
	templates.Toast("Document deleted", "success").Render(r.Context(), w)
}
