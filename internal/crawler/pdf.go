package crawler

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
)

// ExtractPDFText extracts text from a PDF using pdftotext (poppler).
func ExtractPDFText(ctx context.Context, pdfData []byte) (string, error) {
	// Write to temp file
	tmp, err := os.CreateTemp("", "crawl-*.pdf")
	if err != nil {
		return "", fmt.Errorf("creating temp file: %w", err)
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(pdfData); err != nil {
		tmp.Close()
		return "", fmt.Errorf("writing temp file: %w", err)
	}
	tmp.Close()

	// Run pdftotext
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "pdftotext", "-layout", "-enc", "UTF-8", tmp.Name(), "-")
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pdftotext: %w: %s", err, stderr.String())
	}

	return normalizeWhitespace(stdout.String()), nil
}
