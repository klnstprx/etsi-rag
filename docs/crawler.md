# Crawler Design
<!-- packages: internal/crawler, internal/objectstore -->
<!-- related: docs/uma-site-structure.md, docs/architecture.md -->

BFS web crawler for ETSI Informatica (UMA) content acquisition. Fetches HTML pages and PDFs, extracts content, chunks and embeds into the RAG pipeline.

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

1. **Page header/footer stripping**: Lines that repeat across 2+ pages (e.g., "GUÍA DOCENTE DE LA ASIGNATURA") are removed. "Página X de Y" footers are stripped via regex.
2. **Heading detection**: ALL-CAPS lines matching configured `heading_patterns` are converted to `## Title Case Heading` markdown headings. This enables the `StructureAwareChunker` to split PDFs on section boundaries with breadcrumb context.
3. **Normalization**: `normalizeStructured()` collapses excessive blank lines while preserving paragraph boundaries.

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

## Per-Page Language Detection

Document language is determined per page, not per source:
1. `ParseHTML` reads the `lang` attribute from the first `<html>` element.
2. The value is normalised — trim whitespace, lowercase, drop the region subtag (`en-US` → `en`, `es-ES` → `es`).
3. If the normalised value names a language supported by `domain.Language.Valid()` (currently `es` or `en`), it propagates to the document; otherwise it's discarded.
4. On discard — or on PDFs, where no `<html lang>` exists — the embed stage falls back to the source's configured language (`CrawlSource.Language`).

So an `/en/` translation served under a Spanish-language source is correctly tagged `en` if its `<html lang="en">` is present, instead of inheriting the source's `es`. The tag shows up in `documents.language` and downstream in the RAG language filter.

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
| `pdf.go` | PDF -> markdown via pdftotext + pdfinfo + heading detection |
| `filter.go` | Content quality filtering (`isLowQualityContent`) + subject URL dedup (`subjectCode`) |
| `categorizer.go` | URL-based document categorization + content-hash-based `StableDocID` |

## System Dependencies

- `poppler` — provides `pdftotext` (`paru -S poppler`)
- `chromium` or `google-chrome` — headless browser for JS pages (auto-detected by go-rod)
