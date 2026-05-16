# ADR-003: Token-Aware Chunking

## Status
Accepted

## Context
Text must be split into chunks for embedding. Character-based splitting doesn't align with LLM token boundaries, leading to unpredictable chunk sizes in token space.

## Decision
`StructureAwareChunker` as primary chunker: splits markdown on heading boundaries, preserves section hierarchy as `SectionTitle` breadcrumbs (e.g., "Plan > Primer curso"). When a section exceeds the token budget it falls through to an in-package token-based splitter (`splitAndMerge` + `hardSplitByTokens`); when a document has no headings at all, the entire document is run through the same splitter as a fallback.

`Chunk.Content` stores plain body text (with its own `#`-style heading markdown). The breadcrumb lives only on `Chunk.SectionTitle`. A single helper, `chunking.EmbeddingText(breadcrumb, body)`, returns `breadcrumb + ":\n" + body` (or `body` alone when the breadcrumb is empty); the chunker uses it for sizing and the retrieval pipeline uses it for the actual embed-time payload, so the two paths can't drift apart. The FTS5 trigger over `chunks_fts` concatenates `coalesce(section_title, '') || ' ' || content` to mirror the same shape so BM25 sees heading words. Sizing accounts for both the body and the breadcrumb so the embed-time payload still fits the embedding model's input window.

Token counting via `tokencount.Counter` interface:
- `LlamaCppCounter`: HTTP call to llama.cpp `/tokenize` endpoint (precise)
- `TiktokenCounter`: tiktoken-go library (offline, fast)

## Configuration
```toml
[chunking]
max_chunk_size = 256  # tokens
chunk_overlap = 32    # tokens
min_chunk_size = 64   # tokens
```

## Consequences
- Chunk sizes are predictable in token space
- Structure-aware splitting preserves semantic boundaries (headings, sections)
- `SectionTitle` is the single source of truth for the breadcrumb; the LLM-facing context block (`rag.BuildContext`, `agent.read_document`) renders it once in the header and never sees it twice
- `context.Context` required in chunker interface (HTTP-based token counting)
- Markdown pipeline: parser emits markdown -> chunked as markdown -> embedded as markdown
- The FTS5 trigger over `chunks_fts` is the load-bearing piece that keeps heading words visible to BM25; if it ever drops out of sync with the chunker's storage representation, lexical heading-word recall regresses silently. `make reset-db` + re-crawl is the standard playbook for trigger changes.
