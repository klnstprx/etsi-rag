// Package prompts provides embedded system prompt templates.
package prompts

import (
	"bytes"
	_ "embed"
	"fmt"
	"text/template"
)

//go:embed agent.tmpl
var Agent string

// PromptData holds dynamic data available for template rendering.
type PromptData struct {
	DocumentCount int
	ChunkCount    int
}

// RenderAgent renders the agent system prompt. If customTemplate is empty,
// the embedded default template is used.
func RenderAgent(customTemplate string, data PromptData) (string, error) {
	tmpl := customTemplate
	if tmpl == "" {
		tmpl = Agent
	}
	return Render(tmpl, data)
}

// Render parses tmplStr as a text/template and executes it with the given data.
func Render(tmplStr string, data PromptData) (string, error) {
	t, err := template.New("prompt").Parse(tmplStr)
	if err != nil {
		return "", fmt.Errorf("parsing prompt template: %w", err)
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("executing prompt template: %w", err)
	}
	return buf.String(), nil
}
