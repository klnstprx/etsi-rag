# Crawler Design
<!-- packages: internal/crawler, internal/objectstore -->
<!-- related: docs/uma-site-structure.md, docs/architecture.md, docs/configuration.md -->

BFS web crawler for ETSI Informatica (UMA) content acquisition. Fetches HTML pages and PDFs, extracts content, chunks and embeds into the RAG pipeline.

## Sources

Seed URLs live in `config/sources.toml` (single source of truth) — see [configuration.md](configuration.md#crawl-sources) for the schema. The crawler reads the live snapshot from a shared `sources.Registry`; the admin UI mutates the file and calls `Registry.Replace` so the next crawl run sees the change without a server restart. Per-URL state (`crawl_pages`) and per-run history (`crawl_runs`) reference each source by its stable `key` slug.

Each source also carries a `description` and a list of `tags`. They are surfaced in the admin Documents view (joined through `crawl_pages`) for operator filtering, and remain in the schema as forward-compatibility for a future agent-facing scoping tool. A previous `list_sources` tool that exposed them to the agent was reverted on 2026-05-09 after an accuracy regression on small models; today they are not visible to the agent and are not used as SQL filters at ingest or retrieval time.

### Deletion semantics

Removing a source is non-destructive by design — TOML edits should never silently drop indexed content. The two paths converge on the same flow:

- **Admin UI delete** (`DELETE /admin/crawler/sources/{key}`): writes the new `sources.toml`, then calls `MarkPagesStaleBySourceKey` so the source's `crawl_pages` flip from `fetched`/`failed` to `stale`. The toast reports the count.
- **Hand-edit to `sources.toml`**: caught at next boot — `server.New` calls `MarkOrphanPagesStale` with the loaded keys, marking any page whose `source_key` is no longer in the file.

In both cases the embeddings, chunks and stored objects stay in place until the operator clicks **Purge Stale** on the crawler page, which runs `PurgeStaleDocuments` (cascades to chunks/embeddings via FK) and clears `doc_id` on the now-orphaned stale pages. A document that a non-stale page still shares through content dedup is kept. A stale page that a later crawl finds again is re-ingested, even with unchanged content, because it no longer has a document. This gives a typo'd or temporary source removal a recovery window: re-add the key before purging and the next crawl re-uses the existing rows.

## Pipeline

Three-stage concurrent pipeline using Go channels:

```
[Stage 1: Crawl]  -->  [Stage 2: Parse]  -->  [Stage 3: Embed]
   BFS traversal        Content extraction      Chunk + store
   Rate-limited HTTP    go-readability (HTML)    ingester.Ingest()
   chromedp (JS pages)  pdftotext (PDF)          SQLite + RustFS
```

Each source runs its own BFS goroutine. Sources run concurrently, bounded by `config.Concurrency` (channel semaphore), and share one fetcher whose rate limiting is per host (see [Fetching](#fetching)).

Stage concurrency is configurable: `parse_workers` (default 4) controls parallel parse goroutines, `embed_workers` (default 1) controls parallel embed goroutines. The embed stage aborts the crawl after `max_embed_fails` (default 3) consecutive failures to prevent wasting resources on a down embedding backend. A tokenizer that is unreachable or keeps answering 5xx (`tokencount.ErrBackend`) counts as a failure too; a chunking error specific to one document does not. The aborted run is recorded in `crawl_runs` as `failed` with the reason in `error_message`, and `RunAll` returns that error; a run stopped by the operator or by shutdown is `cancelled`.

A run starts only after the embedder answers a ping, the chunker's tokenizer answers, and the object-store bucket answers a HEAD request. Otherwise it returns an error before any `crawl_runs` row exists, instead of crawling without raw bodies or images.

`Ingest` embeds a document's chunks before it writes the `documents` row, and deletes the row again if storing the chunks fails. A failed ingest therefore never leaves a document without chunks, which later pages with the same content would otherwise adopt through content dedup.

## Fetching

Two fetch modes, selected by URL pattern matching against `js_patterns` config:

- **Static**: Standard `http.Client` with configurable timeout and body size limit
- **Dynamic**: Headless Chrome via `chromedp` for JS-rendered pages (e.g., `/centers/subject/` pages load content via AJAX)

The `Browser` type wraps chromedp: one Chrome process per crawler and one tab per fetch, closed after the fetch, bounded by `request_timeout` and the crawl context. A page is ready after its load event once two DOM snapshots taken 500 ms apart differ by at most 10% (longest common subsequence of the snapshot string tables), the rule go-rod's `WaitDOMStable` applied. The tab emulates the 1280x800 landscape laptop, user agent and `Accept-Language: en` that go-rod used, so renders match the existing corpus. If Chrome dies (crash, OOM kill), the next fetch relaunches it, and a fetch that was in flight when it died is tried once more on the new Chrome. A nil `*Browser` in the fetcher means JS rendering unavailable — falls back to static HTTP transparently.

Both modes treat the HTTP status of the page itself as the result: the static path accepts only 200, the browser path any 2xx status of the main document response (`chromedp.NavigateResponse`), so an error page is never parsed as content. UMA's CMS also serves its error template ("Error - Universidad de Málaga", `httperror-description`) with status 200, for example for retired subjects; both modes fail such a page without a retry. The static path takes the content type from the body and the `Content-Type` header: a body starting with the `%PDF-` header is a PDF whatever its declared type, any other body must be declared (or, with no header, sniffed as) `text/html` or `application/xhtml+xml`, and everything else (certificates, configuration profiles, plain text) fails the page.

Rate limiting is per lane and shared by every source. A lane is a host, paced at `request_delay`, or the URLs of one host whose path contains a `throttled_paths` pattern, paced at `throttled_path_delay`; requests in a lane start at least that delay apart, whichever source sends them. A 429 or 503 holds the whole lane, so every source pauses its requests to that lane, for the `Retry-After` header (delta-seconds or HTTP-date) when present, else `rate_limit_backoff` doubled per retry (30 s, 60 s, 120 s, ...), never longer than `rate_limit_max_wait`; the page is then retried, and it fails after `rate_limit_retries` retries with an error naming the status and the attempts. Each wait is logged at warn with the URL, status and duration, and a cancelled crawl ends it at once. Any other non-success status fails the page without a retry. Rate limiting was measured on www.uma.es: a full recrawl from one IP at about 2 requests per second got 429 on a large share of pages, at a rate that kept rising, while a single request a minute later succeeded, so the default `request_delay` of 1 s caps page requests to one host at half that rate. On 2026-11-18 www.uma.es started answering file-browser downloads (`/navegador_de_ficheros/`) with HAProxy 429s (no `Retry-After`) after a burst of about ten, allowing about 1.4 a minute from one IP, while its pages stayed unthrottled at 1 request a second; hence the default `throttled_paths = ['/navegador_de_ficheros/']` paced at 45 s, so the downloads stay under that limit and their backoffs never hold the pages. Subresources Chrome loads for a JS page (scripts, styles, AJAX calls) are not paced.

Raw bytes are stored in RustFS before extraction (`crawl/{html|pdf}/{url_sha256}`), enabling re-processing without re-crawling.

## Content Extraction

- **HTML**: links come first, from the full DOM (nav and sidebar included). Boilerplate is then removed from the DOM: `script`/`style`/`nav`/`header`/`footer`, modals (the hidden iDUMA "Sesión terminada" dialog on every UMA page), cookie banners and the CMS block placeholders ("beforecontenttitle", "Trozos html editables"). The cleaned DOM goes to `readeck/go-readability` and to a content-area walker (`#main_content`, `main`, `article`, ...); the longer markdown wins, and a walk of `<body>` is the fallback when both are empty. A `<br>` or a block boundary separates words; whitespace inside inline markup (`<span>`, `<strong>`, `<a>`) is kept as the HTML has it.
- **PDF**: `pdftotext -enc UTF-8` (reading-order mode) via `poppler`, with post-processing to produce structured markdown. Title comes from the `Title:` line of `pdfinfo` unless it is a placeholder ("subject1", "Document1", "Microsoft Word - ...") or an authoring leftover: a source filename (`*.docx`, `*.xlsx`, `*.rtf`, ...), fewer than 4 letters ("D"), a form label ending in ":" ("Destinatario:"), a line cut after a one- or two-letter word ("ESCUELA TÉCNICA SUPERIOR DE", Roman numerals excepted) or more than 120 characters (a first paragraph cut short). Without a usable Title, a sara.uma.es subject guide (JasperReports, no Title) is named "Guía docente de {Asignatura} ({Grado/Máster en})" from those fields on its first page; any other PDF gets the URL-derived filename title.
- **HTML title**: `og:title` when it is a real title, else readability's title, else the document's own `<title>` (the first one in the HTML namespace, so inline SVG icon titles never win). UMA's CMS stamps the placeholder `og:title` "subject1" on `/centers/subject/` course pages, and readability reads `og:title` too, so the same placeholder check applies to both. A placeholder is a whole default name ("subject1", "Presentación", "PowerPoint Presentation") or a "Microsoft Word - ..." / "Presentación de PowerPoint..." prefix; "Presentación y contacto" is a real title. On a site's root page (one path segment) the CMS sets `og:title` to the portal name ("Grado UMA", "Máster Universitario UMA") and `<title>` to "{site} - {og:title} - Universidad de Málaga"; when the chosen title equals a later ` - ` segment there and is shorter than the first segment, the first segment ("Grado en INGENIERÍA INFORMÁTICA") is used. A longer `og:title` is already the specific name and is kept ("Escuela de Doctorado Universidad de Málaga" rather than "Doctorado").

Parser emits markdown: headings, lists, bold/italic, links. This markdown is stored, embedded, and served to the LLM as-is.

### PDF Post-Processing Pipeline

Raw pdftotext output goes through three stages before ingestion:

1. **Page header/footer stripping**: "Página X de Y" lines are stripped anywhere via regex first, including a total that pdftotext split onto the next line. A running header or footer is then a line found once on, and within the first or last 6 non-empty lines of, at least max(3, 60%) of the pages (e.g., the sara guides' "GUÍA DOCENTE DE LA / ASIGNATURA / Curso 2025 / 2026 / date" block, or a BOE "cve / Verificable en" footer). On each page such lines are dropped only within those edge lines, even when pdftotext puts a body line between a footer and the page edge; deeper in the body they stay. On a page of at most 12 non-empty lines the edge lines cover the whole page, so only the running lines touching its top or bottom are dropped. A line repeated within a page (a table cell such as "6" or "Optativa") never counts as running, and documents under 3 pages have no running lines. Form-feed page boundaries are preserved.
2. **Heading detection + image markers**: ALL-CAPS lines matching configured `heading_patterns` are converted to `## Title Case Heading`. Each page with text emits a single `[[image:p{N}]]` line at its top, N being the physical page number (a blank page emits nothing, and the pages after it keep their numbers, matching the `pdftoppm` render of the same page); the chunker reads these as page boundaries, stamps each emitted chunk with the refs whose range overlaps the chunk's source section (stored on `chunks.image_refs`), and strips the markers from final chunk content. The markers are an internal parser→chunker protocol — never embedded, never user-facing.
3. **Normalization**: `normalizeStructured()` collapses excessive blank lines while preserving paragraph boundaries.

### Image Artifact Rendering

In parallel with text extraction, the parse stage renders one PNG per PDF
page via `pdftoppm -png -r {render_dpi}` and (for HTML pages fetched via
chromedp) one full-page screenshot. Each image is stored in RustFS at
`crawl/img/{url_sha256}/{ref}.png` (`ref` = `p1`, `p2`, … or `full`) and
recorded in the `document_artifacts` SQLite table. See
[ADR-006](decisions/006-multimodal-artifacts.md) for retrieval semantics.

**Coverage caveat:** static-HTML pages (those that don't match a
`js_patterns` prefix) are fetched via plain HTTP, not the browser, so they have
no screenshot artifact. PDFs always get full coverage. Reprocess re-renders
PDFs from archived bytes for free; HTML screenshots require a fresh crawl.

## BFS Algorithm (per source)

1. Start from `src.SeedURL` (normalized), maintain visited set. Every enabled source's seed joins the shared visited set before any source starts, so a seed that other sources link to (the ETSI home page) is fetched once per pass, by the source that owns it.
2. For each URL (bounded by `max_depth`):
   a. Rate-limited fetch -> raw bytes. A redirected page is recorded under its normalized final URL, which is also the base for its relative links; it is skipped when that URL was already seen or is outside the source's scope
   b. Store raw bytes in RustFS
   c. Extract text + links
   d. SHA-256 hash of extracted text without `[[image:...]]` markers (so a screenshot that fails on one fetch does not change the document; `min_content_length` is checked on the same text)
   e. Compare with existing `crawl_pages` record
   f. **Unchanged** (same hash, document still linked): update `last_crawled_at`, skip ingestion, and store any image artifact the document is still missing (one that failed to upload on an earlier crawl)
   g. **Content already seen** (same hash via another URL): upsert a `crawl_pages` row pointing at the existing document, skip re-embedding
   h. **New content**: `ingester.Ingest()` -> upsert `crawl_pages`
   i. **Failure** (fetch, parse, PDF extraction or ingest): `MarkCrawlPageFailed` sets `status='failed'`, the error and `last_crawled_at`. An existing row keeps its `doc_id`, `content_hash` and object key, so a transient failure on a later pass neither orphans the document from the earlier pass nor gets relabelled `stale`. A fetch answered 404 or 410 means the page is gone: the row is still marked `failed`, but its `doc_id` and hash are cleared, so the end-of-run orphan purge deletes the document unless another page shares it.
3. Normalize each discovered link (see below), then enqueue it if it passes `shouldFollow()` for this source and is not yet in the visited set shared by all sources. The scope check runs first, so a link outside this source's scope stays free for the source that owns it
4. After source complete: mark pages not crawled in this run as `status='stale'`; after the run: `PurgeOrphanDocuments()` drops crawled documents no longer referenced by any `crawl_pages`
5. Global `max_pages` counter shared via `atomic.Int64`

## URL Normalization

Every link (seed + discovered) and every redirect target passes through `normalizeURL()` before joining the shared visited set and the BFS queue. It:
- Lowercases the host and puts every `uma.es` host on `https` (UMA redirects `http` to `https`, and its pages carry absolute `http://www.uma.es/...` links), so both schemes collapse to one visited entry and one `documents.url`.
- Strips query-string parameters whose names appear in `[crawler].strip_query_params` (case-insensitive). Defaults cover common tracking params (`utm_*`, `fbclid`, `gclid`, `_ga`, `ref`) and the UMA CMS language switchers (`lang`, `set_language`).
- Canonicalises the remaining query by sorting keys alphabetically, so `?b=2&a=1` and `?a=1&b=2` collapse to one visited entry.
- Leaves malformed URLs untouched (returned as-is) to avoid disrupting caller flow.

Fragment stripping happens earlier in `resolveLink()` (fragments never round-trip to the server). Trailing-slash variants are *not* normalized — the content-addressed doc ID (see below) already prevents duplicate documents from slash variants; we only pay the cost of a wasted fetch.

## Link Following Rules

- `www.uma.es` paths: source's seed path prefix, `/etsi-informatica/`, `/centers/subject/`, `/media/files/`, `/media/tinyimages/file/` (degree verification memorias)
- `sara.uma.es` paths: `ProgramasAsignaturas*.pdf` (guias docentes)
- **Skip**: any `www.uma.es` path with a `noticias`, `news` or `newsitem` segment, under every prefix (time-sensitive, handled by agent tool). This includes the ETSI file browser's `Noticias` folder of news attachments (`/etsi-informatica/navegador_de_ficheros/docs/{listado,descargar}/Noticias/`); a file whose name only mentions news, such as a degree plan's `Anexo I. Noticias relevantes.pdf`, is still followed
- **Subject dedup**: `/centers/subject/` URLs are deduplicated by plan and subject code (`{plan}/{code}`, also in the `/centers/subject/{center}/{plan}/{code}/{year}` form). The first page kept for a plan's subject wins, so a URL that fails, or redirects to a URL already seen or out of scope, leaves the subject's other URLs in the queue. The same subject under another plan is crawled too, because each plan's page links its own guía PDF
- **Reject**: everything else (including external microsites)

## Content Quality Filtering

The parse stage runs `isLowQualityContent()` before ingestion. Pages are rejected if:

- **Content too short**: below `min_content_length` config (default 200 chars). Catches empty stubs and untitled pages.
- **Matches a rejection pattern**: case-insensitive substring match against `[crawler].rejection_patterns`. Ships with defaults for the CMS error page ("No ha sido posible cargar esta página"), soft 404s ("Página no encontrada", "Error 404"), under-construction stubs ("Página en construcción", "Sitio en construcción"), and CMS placeholders ("Sin descripción disponible"). Extendable via TOML or the admin UI config editor.
- **Empty filter pages**: CMS pages (<200 chars) starting with "Filtrar por curso:" and containing no actual data.

Rejected pages increment the `Skipped` progress counter and are logged at debug level.

## Content Freshness

1. **On crawl**: content hash comparison skips unchanged pages
2. **After crawl**: unvisited pages marked `status='stale'`
3. **Admin purge**: remove stale docs from RAG index via admin UI

## Reprocess

`POST /admin/crawler/reprocess` re-runs the parse and embed stages over every `crawl_pages` row in `fetched` status, and every `failed` row that still links the document of its last good crawl, replaying the bytes archived in RustFS — no network fetch. Use it when chunking, parsing, or embedding code/config changes but the crawled HTML/PDF is still fresh.

Reprocess sets the embed stage's `force` flag, which changes these things vs. a regular crawl:

- The "content hash unchanged" early-out is skipped, so identical bytes still flow through chunking and embedding.
- When the content-addressed `doc_id` already exists in `documents`, the stale row is dropped via `DeleteDocument` (chunks and embeddings cascade) and the document is re-ingested with the current embedder/chunker. Without this delete, `ingester.Ingest` would hit the `documents.id` PK and reprocess would silently skip everything.
- The page keeps the `status`, error and `last_crawled_at` of its last crawl, since reprocess fetches nothing: a `failed` page stays `failed` with its error.
- The re-ingested document keeps the more **canonical** of its current `documents.url` (with its title) and the page being reprocessed (`preferCanonical`, same rule as the live-crawl dedup path). UMA serves identical content at the bare program URL (`/grado-en-X/`) and its `/cms/menu/...` sub-pages, which collapse to one `doc_id`; without this, recreating under whichever colliding page is reprocessed last would flip `documents.url` off the bare URL and silently break lookups that key on it (e.g. eval `golden_urls`).

This is the supported path after switching embedding model or dimension: save the new model/dimension in the admin config editor, then click **Reprocess**. The admin UI shows a warning toast on the embedding-section save when either field changed.

## Per-Page Language Detection

Document language is determined per page:
1. `ParseHTML` reads the `lang` attribute from the first `<html>` element.
2. The value is normalised — trim whitespace, lowercase, drop the region subtag (`en-US` → `en`, `es-ES` → `es`).
3. If the normalised value names a language supported by `domain.Language.Valid()` (currently `es` or `en`), it propagates to the document; otherwise it's discarded.
4. On discard — or on PDFs, where no `<html lang>` exists — the embed stage falls back to `domain.LanguageES` (Spanish), since the project crawls UMA-only content.

So an `/en/` translation served under a Spanish-language source is correctly tagged `en` if its `<html lang="en">` is present. The tag is persisted in `documents.language` and surfaced in the `/v1/rag/documents` listing for metadata, but is no longer used to filter retrieval — multilingual embeddings made the hard filter unnecessary, and it was removed in commit `81d3b12`.

## Content-Addressed Document IDs

Document IDs are derived from the SHA-256 content hash, not the URL: `doc_id = "crawl-" + sha256(extracted_text)[:24]`, the text taken without its image markers. Two URLs that yield identical extracted text map to the same `doc_id` — so the same content is stored, chunked, and embedded only once, and both URLs become rows in `crawl_pages` pointing at the shared document.

Consequences:
- **`crawl_pages.doc_id` is many-to-one with `documents.id`** (multiple URLs can share a doc).
- **When a URL's content changes**, its `crawl_pages.doc_id` points at a new document; the old one may remain if another URL still references it, or become an orphan.
- **Orphan cleanup**: `PurgeOrphanDocuments` runs automatically at the end of every `runCrawl` / `runReprocess`. It deletes crawled documents (`id LIKE 'crawl-%'`) whose `id` is not in `SELECT doc_id FROM crawl_pages`; documents ingested through the API never have a crawl page and are left alone. Chunks and embeddings cascade via `ON DELETE CASCADE`.
- **`documents.url`** stores a representative URL for the content. Among URLs that collide on the same `doc_id`, `preferCanonical` picks the most canonical one (fewest path segments, then shortest path, then `https`, then no query string) rather than whichever won the timing race — on both live crawl and reprocess. `documents.title` moves with the URL on both paths, so the title is the canonical page's. The full list of URLs surfacing a given document lives in `crawl_pages`.

## Files

| File | Purpose |
|------|---------|
| `crawler.go` | Orchestration: `RunAll`, `Reprocess`, `Cancel`, `IsRunning`; one `pipeline` driver shared by crawl and reprocess |
| `stages.go` | Stage workers: crawl (BFS + link `frontier`), replay (reprocess feed), parse, embed |
| `pipeline.go` | Items passed between stages and the live `RunProgress` counters |
| `fetcher.go` | Per-host pacing + body size limit + content-type check + JS pattern routing + 429/503 backoff and retry |
| `browser.go` | chromedp headless Chrome wrapper: readiness wait, full-page screenshot, relaunch after Chrome dies |
| `parser.go` | HTML -> markdown + links (readability + html walker) |
| `pdf.go` | PDF -> markdown via pdftotext + pdfinfo + heading detection; per-page PNG renders via pdftoppm |
| `filter.go` | Link following (`shouldFollow`, blocked extensions), content quality filtering (`isLowQualityContent`), subject URL dedup (`subjectCode`) |
| `canonical.go` | `preferCanonical`: which of two URLs sharing a document keeps `documents.url` |

## System Dependencies

- `poppler` — provides `pdftotext`, `pdfinfo`, `pdftoppm` (`paru -S poppler`)
- `chromium` or `google-chrome` — headless browser for JS pages (found on `$PATH` or in the standard install locations by chromedp; never downloaded)
