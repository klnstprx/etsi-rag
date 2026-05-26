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
//
// requestTimeout bounds every handler — including the agent tool-calling
// loop on /v1/chat/completions, which on slow local LLMs can run for several
// minutes. Pair with a server.write_timeout that outlasts it (see
// config.applyDefaults) so the handler timeout fires first.
func NewRouter(handler *Handler, logger zerolog.Logger, requestTimeout time.Duration) *chi.Mux {
	SetLogger(logger)
	r := chi.NewRouter()

	// Middleware stack
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(zerologMiddleware(logger))
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(requestTimeout))

	// Health check (outside /v1 for simplicity)
	r.Get("/health", handler.HealthCheck)

	// OpenAI-compatible API routes
	r.Route("/v1", func(r chi.Router) {
		// Health check (llama-server also exposes /v1/health)
		r.Get("/health", handler.HealthCheck)

		// Standard OpenAI endpoint — strictly OpenAI-shaped (no RAG extras).
		r.Post("/chat/completions", handler.ChatCompletions)

		// Proprietary agent endpoint — same agent, but the response carries the
		// full RAG context (citations, images, tool calls) the OpenAI endpoint
		// omits. Consumed by the admin UI and the eval harness.
		r.Post("/agent/chat", handler.AgentChat)

		// Image artifact serving — fed to vision-capable LLMs as image_url.
		// {token} is the artifact's opaque public_token; .png suffix is
		// optional and ignored by the handler.
		r.Get("/artifacts/{token}", handler.GetArtifact)

		// RAG endpoints
		r.Route("/rag", func(r chi.Router) {
			r.Post("/ingest", handler.Ingest)
			r.Get("/documents", handler.ListDocuments)
			r.Delete("/documents/{id}", handler.DeleteDocument)
			r.Post("/query", handler.RAGQuery)
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
