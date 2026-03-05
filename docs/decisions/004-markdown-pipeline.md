# ADR-004: Markdown Pipeline

## Status
Accepted

## Context
Crawled content needs to be stored, chunked, and served to the LLM. Options: plain text, HTML, or markdown.

## Decision
Full markdown pipeline. The HTML parser emits markdown (headings, lists, bold/italic, links). Content is stored as markdown, chunked as markdown, embedded as markdown, and served to the LLM as markdown.

## Consequences
- LLMs understand markdown natively — better comprehension than plain text
- Structure preserved through the pipeline (headings enable structure-aware chunking)
- Smaller than HTML, more structured than plain text
- go-readability extracts main content, then custom formatting converts to markdown
