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

## Fetching

Two fetch modes, selected by URL pattern matching against `js_patterns` config:

- **Static**: Standard `http.Client` with configurable timeout and body size limit
- **Dynamic**: Headless Chrome via `go-rod/rod` for JS-rendered pages (e.g., `/centers/subject/` pages load content via AJAX)

The `Browser` type wraps rod. A nil `*Browser` in the fetcher means JS rendering unavailable — falls back to static HTTP transparently.

Raw bytes are stored in RustFS before extraction (`crawl/{html|pdf}/{url_sha256}`), enabling re-processing without re-crawling.

## Content Extraction

- **HTML**: `go-shiori/go-readability` for main content extraction (strips boilerplate). Link extraction uses separate `golang.org/x/net/html` walker (needs full DOM including nav/sidebar).
- **PDF**: `pdftotext -layout -enc UTF-8` via `poppler` system package.

Parser emits markdown: headings, lists, bold/italic, links. This markdown is stored, embedded, and served to the LLM as-is.

## BFS Algorithm (per source)

1. Start from `src.SeedURL`, maintain visited set
2. For each URL (bounded by `max_depth`):
   a. Rate-limited fetch -> raw bytes
   b. Store raw bytes in RustFS
   c. Extract text + links
   d. SHA-256 hash of extracted text
   e. Compare with existing `crawl_pages` record
   f. **Unchanged**: update `last_crawled_at`, skip ingestion
   g. **New/changed**: delete old doc -> `pipeline.Ingest()` -> upsert `crawl_pages`
3. Enqueue discovered links passing `shouldFollow()`
4. After source complete: mark unvisited pages as `status='stale'`
5. Global `max_pages` counter shared via `atomic.Int64`

## Link Following Rules

- `www.uma.es` paths: source's seed path prefix, `/etsi-informatica/`, `/centers/subject/`, `/media/files/`
- `sara.uma.es` paths: `ProgramasAsignaturas*.pdf` (guias docentes)
- **Skip**: `noticias`, `news`, `newsitem` (time-sensitive, handled by agent tool)
- **Reject**: everything else (including external microsites)

## Content Freshness

1. **On crawl**: content hash comparison skips unchanged pages
2. **After crawl**: unvisited pages marked `status='stale'`
3. **Admin purge**: remove stale docs from RAG index via admin UI

## Files

| File | Purpose |
|------|---------|
| `crawler.go` | Orchestration: `RunAll`, `RunSource`, `Cancel`, `IsRunning` |
| `stages.go` | Three-stage pipeline wiring |
| `fetcher.go` | Rate-limited HTTP + body size limit + JS pattern routing |
| `browser.go` | go-rod headless Chrome wrapper |
| `parser.go` | HTML -> markdown + links (readability + html walker) |
| `pdf.go` | PDF -> text via pdftotext |

## System Dependencies

- `poppler` — provides `pdftotext` (`paru -S poppler`)
- `chromium` or `google-chrome` — headless browser for JS pages (auto-detected by go-rod)
