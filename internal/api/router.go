package api

import (
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/rs/zerolog"
)

// NewRouter creates a new Chi router with middleware and routes.
// The API follows OpenAI conventions where applicable.
func NewRouter(handler *Handler, logger zerolog.Logger) *chi.Mux {
	r := chi.NewRouter()

	// Middleware stack
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(zerologMiddleware(logger))
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(120 * time.Second)) // LLM responses can be slow

	// Health check (outside /v1 for simplicity)
	r.Get("/health", handler.HealthCheck)

	// OpenAI-compatible API routes
	r.Route("/v1", func(r chi.Router) {
		// Health check (llama-server also exposes /v1/health)
		r.Get("/health", handler.HealthCheck)

		// Standard OpenAI endpoints (proxied to llama-server or handled locally)
		r.Post("/chat/completions", handler.ChatCompletions)
		r.Post("/embeddings", handler.Embeddings)

		// RAG endpoints
		r.Route("/rag", func(r chi.Router) {
			r.Post("/ingest", handler.Ingest)
			r.Get("/documents", handler.ListDocuments)
			r.Delete("/documents/{id}", handler.DeleteDocument)
		})
	})

	return r
}

// zerologMiddleware creates a request logging middleware using zerolog.
func zerologMiddleware(logger zerolog.Logger) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)

			// Use Debug level for noisy polling/static paths
			lvl := zerolog.InfoLevel
			p := r.URL.Path
			if p == "/admin/crawler/status" || strings.HasPrefix(p, "/admin/static/") {
				lvl = zerolog.DebugLevel
			}

			logger.WithLevel(lvl).
				Str("method", r.Method).
				Str("path", p).
				Int("status", ww.Status()).
				Int("bytes", ww.BytesWritten()).
				Dur("duration", time.Since(start)).
				Str("request_id", middleware.GetReqID(r.Context())).
				Msg("request")
		})
	}
}
