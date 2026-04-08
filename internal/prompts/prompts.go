// Package prompts provides embedded system prompts loaded from markdown files.
package prompts

import _ "embed"

//go:embed agent.md
var Agent string
