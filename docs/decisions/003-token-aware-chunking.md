# ADR-003: Token-Aware Chunking

## Status
Accepted

## Context
Text must be split into chunks for embedding. Character-based splitting doesn't align with LLM token boundaries, leading to unpredictable chunk sizes in token space.

## Decision
`StructureAwareChunker` as primary chunker: splits markdown on heading boundaries, preserves section hierarchy as `SectionTitle` breadcrumbs (e.g., "Plan > Primer curso"). Falls back to `RecursiveTokenChunker` for content within sections or unstructured text.

Token counting via `tokencount.Counter` interface:
- `LlamaCppCounter`: HTTP call to llama.cpp `/tokenize` endpoint (precise)
- `TiktokenCounter`: tiktoken-go library (offline, fast)
- `EstimateCounter`: `len(text)/4` (no dependencies, rough)

## Configuration
```toml
[chunking]
chunk_size = 256     # tokens
chunk_overlap = 32   # tokens
min_chunk_size = 64  # tokens
```

## Consequences
- Chunk sizes are predictable in token space
- Structure-aware splitting preserves semantic boundaries (headings, sections)
- `SectionTitle` provides context without duplicating content
- `context.Context` required in chunker interface (HTTP-based token counting)
- Markdown pipeline: parser emits markdown -> chunked as markdown -> embedded as markdown
