package ingest

import "errors"

// ErrChunking sentinels chunking-stage failures during Ingest so callers can
// distinguish them from embedding or storage failures (for circuit-breaker
// scoping, retry policy, etc.). Check with errors.Is.
var ErrChunking = errors.New("chunking failed")
