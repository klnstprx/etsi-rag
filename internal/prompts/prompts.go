// Package prompts provides embedded system prompts loaded from markdown files.
package prompts

import _ "embed"

//go:embed rag_es.md
var RAGEs string

//go:embed rag_en.md
var RAGEn string

//go:embed agent.md
var Agent string
