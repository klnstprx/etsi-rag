# ADR-006: Multimodal Image Artifacts via Inline Markers

## Status
Accepted — implementation evolved 2026-05-09 (see [Update](#update-2026-05-09-chunker-side-stamping) at the end of this file).

## Context
Text-only RAG can't answer questions about layout-heavy documents (UMA
schedules, plan-de-estudios grids, scanned tables) or "can you open this
PDF?". The crawler already archives raw HTML/PDF bodies in RustFS; we want to
turn those archives into image artifacts the agent can attach to a
vision-capable LLM.

Three structural questions:

1. How do chunks reference images that belong to the same source?
2. Which message slot carries images on the wire?
3. How is vision capability detected per-backend?

## Decision

**Inline markers in chunk text + a `document_artifacts` table.** Each rendered
image (PDF page or HTML screenshot) gets a row keyed `(doc_id, ref)` where
`ref` is `p1`, `p2`, …, `full`. The parser emits `[[image:p1]]` markers at
PDF page boundaries (and `[[image:full]]` at the top of HTML pages with a
screenshot). Markers survive chunking unaltered; the embedder strips them
in-flight (so they don't pollute the embedding space) but the storage path
keeps them, so retrieval can scan a chunk's text for markers and resolve them
to artifact URLs.

Rejected alternatives:
- Page-number column on chunks: invasive schema change to `chunks`, doesn't
  cover HTML, requires per-page chunking discipline.
- Per-figure cropping (`pdfimages`, bbox detection): heavier dep stack;
  whole-page renders capture tables/figures implicitly. Deferred.

**Marker syntax `[[image:ref]]`**, not `<|image:ref|>`. The bracketed form
avoids collisions with tokenizer special tokens (`<|im_start|>`,
`<|endoftext|>`, etc.) that some embedders may strip or mishandle.

**Synthetic follow-up user message for vision injection.** openai-go v1.12.0
defines `ChatCompletionToolMessageParamContentUnion` as
`{OfString, OfArrayOfContentParts}` where `OfArrayOfContentParts` is
`[]ChatCompletionContentPartTextParam` — text-only. Image content blocks
must live in *user* messages. After each tool call, when the configured
backend supports vision, the agent appends one synthetic user message:

```jsonc
{role: "user", content: [
  {type: "text",      text: "The previous tool returned the following …"},
  {type: "image_url", image_url: {url: "<base>/v1/artifacts/<token>.png"}},
  …
]}
```

The image_url uses an HTTP URL into our public `/v1/artifacts/{token}.png`
route (configurable `[agent].public_base_url`). HTTP URLs over inline data:
URLs avoid the 33% base64 expansion repeated per turn and let the LLM
backend cache fetches.

**Addendum (2026-11): inline is now the default.** llama.cpp's server now
refuses to fetch media from local or private addresses unless started with
`--media-allow-private-urls` (upstream SSRF fix 8954f1948). A self-hosted
deployment's `public_base_url` is such an address, so the request failed with
HTTP 400 and the whole agent turn failed with it. `[agent.vision]
image_transport = 'inline'` (the default) has the app load each artifact
from the object store and send it as a data URI; the backend fetches
nothing, and an image that fails to load is dropped instead of failing the
turn. The cost is the base64 expansion noted above. `image_transport =
'url'` keeps the original behaviour for backends that can reach the app.

Rejected alternatives:
- Image parts inside the tool message (raw JSON shenanigans): non-portable
  across OpenAI-compliant backends.
- Resending the original user turn with images attached: doubles latency
  and complicates the agent loop.
- Inline data: URLs only: works against any backend, but expensive on the
  wire when artifacts are reused across turns.

**Vision capability is a manual flag with a single canonical probe.**
`AgentVisionConfig.Enabled bool` is authoritative — the operator asserts that
the configured `agent.model` accepts `image_url` content blocks. The flag
lives on the agent stage, not the backend, because a single llama-server
router can serve multiple models with different multimodal capabilities;
"backend supports vision" is a category error when only the loaded model
matters. (Original design had this on `BackendConfig.SupportsVision`; moved
2026-05-04 with a hard-break legacy-key migration.)

For llama.cpp backends, `FetchLlamaCppProps` populates
`LlamaCppProps.SupportsVision` from exactly one source: the
`modalities.vision` boolean returned by `GET /props?model=<id>` on recent
llama-server builds. We commit to that shape rather than scanning
speculative field names like `mmproj`, `multimodal`, or CLI args of
`/v1/models[].status.args` — the canonical response is rich and stable, and
defensive multi-field matching just adds noise. The probe never overrides
the manual flag; on disagreement it logs a hint so the user can flip the
toggle in admin without re-reading discovery output. `modalities.audio` is
also recorded for future use.

## Consequences

- **Schema**: one new table (`document_artifacts`) with `(doc_id, ref,
  kind, object_store_key, mime_type, width, height, public_token)`.
  `public_token` is a 128-bit random hex string used in the `/v1/artifacts/`
  route to avoid trivial enumeration of integer IDs.
- **RustFS layout**: extended convention `crawl/img/{url_sha256}/{ref}.png`
  alongside the existing `crawl/{html,pdf}/{url_sha256}` body archive.
- **Crawler**: new `pdftoppm` shell-out (poppler, already a dep alongside
  `pdftotext`). Per-page render failures are logged and skipped; the
  document still ingests with text-only chunks if all renders fail.
- **Reprocess parity**: PDF artifacts re-render from archived bytes for
  free. HTML screenshots are NOT in RustFS (only the original body), so
  reprocess produces no HTML thumbnails — a fresh crawl is required for
  HTML coverage. Acceptable: the user's primary use case is PDFs.
- **Hot-swap**: agent reads `Agent.Vision.Enabled` and
  `Agent.Vision.PublicBaseURL` through a getter (`VisionGetter`) so admin
  toggles take effect without restart. The getter reports the two flags
  independently — `(true, "")` when `Enabled=true` but `PublicBaseURL` is
  empty — so the agent can log the actual misconfiguration instead of
  misattributing it.
- **Backwards compatibility**: every chunk with a marker that has no
  matching artifact row is silently text-only. No errors propagate.

## Update 2026-05-09: chunker-side stamping

The original design above had per-line markers ride through chunking
unchanged, and the embedder stripped them in-flight. In practice this
meant every chunk's text had to be scanned at retrieval time to figure
out which artifacts it implied, and the marker noise complicated the
chunker's section-boundary logic.

Replaced with a parser→chunker handoff:

- The PDF parser emits one `[[image:pN]]` line at the top of each page
  (sparse, no per-line repetition). HTML parsing emits `[[image:full]]`
  once at the top when a browser screenshot was captured.
- `markers.ExtractBoundaries` returns the marker positions alongside
  marker-free text.
- The chunker (`StructureAwareChunker`) walks the boundaries while
  splitting, stamps each emitted chunk with the refs whose page range
  overlaps the chunk's source span, and stores them in a new
  `chunks.image_refs` column (comma-separated `p1,p2,full,…`).
- Chunk content is marker-free from the moment it leaves the chunker;
  the embedder and the agent both consume clean text. Retrieval reads
  `Chunk.ImageRefs` directly to look up `document_artifacts` and build
  the synthetic vision user message.

The on-the-wire vision-injection design (synthetic follow-up user
message with `image_url` parts) and the capability-flag decision
(`agent.vision.enabled`, with `modalities.vision` as a hint) are
unchanged.
