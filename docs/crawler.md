# Crawler Design
<!-- packages: internal/crawler, internal/objectstore -->
<!-- related: docs/uma-site-structure.md, docs/architecture.md, docs/configuration.md -->

BFS web crawler for ETSI Informatica (UMA) content acquisition. Fetches HTML pages and PDFs, extracts content, chunks and embeds into the RAG pipeline.

## Sources

Seed URLs live in `config/sources.toml` (single source of truth) — see [configuration.md](configuration.md#crawl-sources) for the schema. The crawler holds an in-memory snapshot loaded at boot; the admin UI mutates the file and calls `Crawler.SetSources` to hot-reload it for the next run. Per-URL state (`crawl_pages`) and per-run history (`crawl_runs`) reference each source by its stable `key` slug.

Each source also carries a `description` and a list of `tags`. These are agent-facing only — surfaced through the `list_sources` tool so the agent can pass `source_keys` to `semantic_search` for scoped retrieval. They are not stored on documents and not used as SQL filters at ingest or retrieval time.

### Deletion semantics

Removing a source is non-destructive by design — TOML edits should never silently drop indexed content. The two paths converge on the same flow:

- **Admin UI delete** (`DELETE /admin/crawler/sources/{key}`): writes the new `sources.toml`, then calls `MarkPagesStaleBySourceKey` so the source's `crawl_pages` flip from `fetched`/`failed` to `stale`. The toast reports the count.
- **Hand-edit to `sources.toml`**: caught at next boot — `server.New` calls `MarkOrphanPagesStale` with the loaded keys, marking any page whose `source_key` is no longer in the file.

In both cases the embeddings, chunks and stored objects stay in place until the operator clicks **Purge Stale** on the crawler page, which runs `PurgeStaleDocuments` (cascades to chunks/embeddings via FK) and clears `doc_id` on the now-orphaned stale pages. This gives a typo'd or temporary source removal a recovery window: re-add the key before purging and the next crawl re-uses the existing rows.

## Pipeline

Three-stage concurrent pipeline using Go channels:

```
[Stage 1: Crawl]  -->  [Stage 2: Parse]  -->  [Stage 3: Embed]
   BFS traversal        Content extraction      Chunk + store
   Rate-limited HTTP    go-readability (HTML)    pipeline.Ingest()
   go-rod (JS pages)    pdftotext (PDF)          SQLite + RustFS
```

Each source runs its own BFS goroutine with independent rate limiting. Sources run concurrently, bounded by `config.Concurrency` (semaphore via `errgroup`).

Stage concurrency is configurable: `parse_workers` (default 4) controls parallel parse goroutines, `embed_workers` (default 1) controls parallel embed goroutines. The embed stage aborts the crawl after `max_embed_fails` (default 3) consecutive failures to prevent wasting resources on a down embedding backend.

## Fetching

Two fetch modes, selected by URL pattern matching against `js_patterns` config:

- **Static**: Standard `http.Client` with configurable timeout and body size limit
- **Dynamic**: Headless Chrome via `go-rod/rod` for JS-rendered pages (e.g., `/centers/subject/` pages load content via AJAX)

The `Browser` type wraps rod. A nil `*Browser` in the fetcher means JS rendering unavailable — falls back to static HTTP transparently.

Raw bytes are stored in RustFS before extraction (`crawl/{html|pdf}/{url_sha256}`), enabling re-processing without re-crawling.

## Content Extraction

- **HTML**: `go-shiori/go-readability` for main content extraction (strips boilerplate). Link extraction uses separate `golang.org/x/net/html` walker (needs full DOM including nav/sidebar).
- **PDF**: `pdftotext -enc UTF-8` (reading-order mode) via `poppler`, with post-processing to produce structured markdown. Title extracted via `pdfinfo` when available.

Parser emits markdown: headings, lists, bold/italic, links. This markdown is stored, embedded, and served to the LLM as-is.

### PDF Post-Processing Pipeline

Raw pdftotext output goes through three stages before ingestion:

1. **Page header/footer stripping**: Lines that repeat across 2+ pages (e.g., "GUÍA DOCENTE DE LA ASIGNATURA") are removed. "Página X de Y" footers are stripped via regex. Form-feed page boundaries are preserved.
2. **Heading detection + image markers**: ALL-CAPS lines matching configured `heading_patterns` are converted to `## Title Case Heading`. Each page boundary emits a `[[image:p{N}]]` marker so chunks can later reference the corresponding `pdftoppm`-rendered page image.
3. **Normalization**: `normalizeStructured()` collapses excessive blank lines while preserving paragraph boundaries.

### Image Artifact Rendering

In parallel with text extraction, the parse stage renders one PNG per PDF
page via `pdftoppm -png -r {render_dpi}` and (for HTML pages fetched via
go-rod) one full-page screenshot. Each image is stored in RustFS at
`crawl/img/{url_sha256}/{ref}.png` (`ref` = `p1`, `p2`, … or `full`) and
recorded in the `document_artifacts` SQLite table. See
[ADR-006](decisions/006-multimodal-artifacts.md) for retrieval semantics.

**Coverage caveat:** static-HTML pages (those that don't match a
`js_patterns` prefix) are fetched via plain HTTP, not go-rod, so they have
no screenshot artifact. PDFs always get full coverage. Reprocess re-renders
PDFs from archived bytes for free; HTML screenshots require a fresh crawl.

## BFS Algorithm (per source)

1. Start from `src.SeedURL` (normalized), maintain visited set
2. For each URL (bounded by `max_depth`):
   a. Rate-limited fetch -> raw bytes
   b. Store raw bytes in RustFS
   c. Extract text + links
   d. SHA-256 hash of extracted text
   e. Compare with existing `crawl_pages` record
   f. **Unchanged**: update `last_crawled_at`, skip ingestion
   g. **Content already seen** (same hash via another URL): upsert a `crawl_pages` row pointing at the existing document, skip re-embedding
   h. **New content**: `pipeline.Ingest()` -> upsert `crawl_pages`
3. Normalize each discovered link (strip tracking params + sort query keys), then enqueue if it passes `shouldFollow()`
4. After source complete: mark unvisited pages as `status='stale'`; after the run: `PurgeOrphanDocuments()` drops documents no longer referenced by any `crawl_pages`
5. Global `max_pages` counter shared via `atomic.Int64`

## URL Normalization

Every link (seed + discovered) passes through `normalizeURL()` before joining `globalVisited` and the BFS queue. It:
- Strips query-string parameters whose names appear in `[crawler].strip_query_params` (case-insensitive). Defaults cover common tracking params (`utm_*`, `fbclid`, `gclid`, `_ga`, `ref`) and the UMA CMS language switcher (`lang`).
- Canonicalises the remaining query by sorting keys alphabetically, so `?b=2&a=1` and `?a=1&b=2` collapse to one visited entry.
- Leaves malformed URLs untouched (returned as-is) to avoid disrupting caller flow.

Fragment stripping happens earlier in `resolveLink()` (fragments never round-trip to the server). Trailing-slash variants are *not* normalized — the content-addressed doc ID (see below) already prevents duplicate documents from slash variants; we only pay the cost of a wasted fetch.

## Link Following Rules

- `www.uma.es` paths: source's seed path prefix, `/etsi-informatica/`, `/centers/subject/`, `/media/files/`
- `sara.uma.es` paths: `ProgramasAsignaturas*.pdf` (guias docentes)
- **Skip**: `noticias`, `news`, `newsitem` (time-sensitive, handled by agent tool)
- **Subject dedup**: `/centers/subject/{plan}/{code}` URLs are deduplicated by subject code — the first URL seen for a given code wins; same subject under different degree plans is not re-crawled
- **Reject**: everything else (including external microsites)

## Content Quality Filtering

The parse stage runs `isLowQualityContent()` before ingestion. Pages are rejected if:

- **Content too short**: below `min_content_length` config (default 200 chars). Catches empty stubs and untitled pages.
- **Matches a rejection pattern**: case-insensitive substring match against `[crawler].rejection_patterns`. Ships with defaults for iDUMA SSO session-expired pages ("Se ha detectado un cierre de sesión"), soft 404s ("Página no encontrada", "Error 404"), under-construction stubs ("Página en construcción", "Sitio en construcción"), and CMS placeholders ("Sin descripción disponible"). Extendable via TOML or the admin UI config editor.
- **Empty filter pages**: CMS pages (<200 chars) starting with "Filtrar por curso:" and containing no actual data.

Rejected pages increment the `Skipped` progress counter and are logged at debug level.

## Content Freshness

1. **On crawl**: content hash comparison skips unchanged pages
2. **After crawl**: unvisited pages marked `status='stale'`
3. **Admin purge**: remove stale docs from RAG index via admin UI

## Reprocess

`POST /admin/crawler/reprocess` re-runs the parse and embed stages over every `crawl_pages` row in `fetched` status, replaying the bytes archived in RustFS — no network fetch. Use it when chunking, parsing, or embedding code/config changes but the crawled HTML/PDF is still fresh.

Reprocess sets the embed stage's `force` flag, which changes two things vs. a regular crawl:

- The "content hash unchanged" early-out is skipped, so identical bytes still flow through chunking and embedding.
- When the content-addressed `doc_id` already exists in `documents`, the stale row is dropped via `DeleteDocument` (chunks and embeddings cascade) and the document is re-ingested with the current embedder/chunker. Without this delete, `pipeline.Ingest` would hit the `documents.id` PK and reprocess would silently skip everything.

This is the supported path after switching embedding model or dimension: save the new model/dimension in the admin config editor, then click **Reprocess**. The admin UI shows a warning toast on the embedding-section save when either field changed.

## Per-Page Language Detection

Document language is determined per page:
1. `ParseHTML` reads the `lang` attribute from the first `<html>` element.
2. The value is normalised — trim whitespace, lowercase, drop the region subtag (`en-US` → `en`, `es-ES` → `es`).
3. If the normalised value names a language supported by `domain.Language.Valid()` (currently `es` or `en`), it propagates to the document; otherwise it's discarded.
4. On discard — or on PDFs, where no `<html lang>` exists — the embed stage falls back to `domain.LanguageES` (Spanish), since the project crawls UMA-only content.

So an `/en/` translation served under a Spanish-language source is correctly tagged `en` if its `<html lang="en">` is present. The tag is persisted in `documents.language` and surfaced in the `/v1/rag/documents` listing for metadata, but is no longer used to filter retrieval — multilingual embeddings made the hard filter unnecessary, and it was removed in commit `81d3b12`.

## Content-Addressed Document IDs

Document IDs are derived from the SHA-256 content hash, not the URL: `doc_id = "crawl-" + sha256(extracted_text)[:24]`. Two URLs that yield identical extracted text map to the same `doc_id` — so the same content is stored, chunked, and embedded only once, and both URLs become rows in `crawl_pages` pointing at the shared document.

Consequences:
- **`crawl_pages.doc_id` is many-to-one with `documents.id`** (multiple URLs can share a doc).
- **When a URL's content changes**, its `crawl_pages.doc_id` points at a new document; the old one may remain if another URL still references it, or become an orphan.
- **Orphan cleanup**: `PurgeOrphanDocuments` runs automatically at the end of every `runCrawl` / `runReprocess`. It deletes documents whose `id` is not in `SELECT doc_id FROM crawl_pages`; chunks and embeddings cascade via `ON DELETE CASCADE`.
- **`documents.url`** stores a representative URL (first seen for this content). The full list of URLs surfacing a given document lives in `crawl_pages`.

## Files

| File | Purpose |
|------|---------|
| `crawler.go` | Orchestration: `RunAll`, `RunSource`, `Cancel`, `IsRunning` |
| `stages.go` | Three-stage pipeline wiring |
| `fetcher.go` | Rate-limited HTTP + body size limit + JS pattern routing |
| `browser.go` | go-rod headless Chrome wrapper |
| `parser.go` | HTML -> markdown + links (readability + html walker) |
| `pdf.go` | PDF -> markdown via pdftotext + pdfinfo + heading detection; per-page PNG renders via pdftoppm |
| `filter.go` | Content quality filtering (`isLowQualityContent`) + subject URL dedup (`subjectCode`) |
| `categorizer.go` | URL-based document categorization + content-hash-based `StableDocID` |

## System Dependencies

- `poppler` — provides `pdftotext`, `pdfinfo`, `pdftoppm` (`paru -S poppler`)
- `chromium` or `google-chrome` — headless browser for JS pages (auto-detected by go-rod)
