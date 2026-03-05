# Go Guidelines
<!-- packages: all -->

Project-specific Go conventions and the Go Proverbs that guide design decisions.

## Go Proverbs

*Simple, Poetic, Pithy*

- Don't communicate by sharing memory, share memory by communicating.
- Concurrency is not parallelism.
- Channels orchestrate; mutexes serialize.
- The bigger the interface, the weaker the abstraction.
- Make the zero value useful.
- interface{} says nothing.
- Gofmt's style is no one's favorite, yet gofmt is everyone's favorite.
- A little copying is better than a little dependency.
- Syscall must always be guarded with build tags.
- Cgo must always be guarded with build tags.
- Cgo is not Go.
- With the unsafe package there are no guarantees.
- Clear is better than clever.
- Reflection is never clear.
- Errors are values.
- Don't just check errors, handle them gracefully.
- Design the architecture, name the components, document the details.
- Documentation is for users.
- Don't panic.

## Project Conventions

### Dependencies

- **No CGO** — `CGO_ENABLED=0` for portable builds. Use `modernc.org/sqlite` (pure Go) instead of `mattn/go-sqlite3`.
- **Standard library first** — prefer stdlib over external deps when reasonable. A little copying is better than a little dependency.
- **Router**: `go-chi/chi` for HTTP routing.
- **Logging**: `zerolog` for structured logging.
- **LLM SDK**: `github.com/openai/openai-go` — use its types throughout, no custom message types.
- **Object storage**: `aws-sdk-go-v2` for S3-compatible (RustFS).

### Interfaces

Keep interfaces small and focused. The `llm.LLM` interface exists for testability — it wraps only what consumers need. The `tokencount.Counter` interface has three implementations (LlamaCpp, Tiktoken, Estimate) selected at runtime.

```go
// Good: small, focused
type LLM interface {
    Embed(ctx context.Context, input []string) ([][]float32, error)
    Chat(ctx context.Context, req openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)
    ChatStream(ctx context.Context, req openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error)
}
```

### Error Handling

- Return errors, don't panic.
- Wrap errors with context: `fmt.Errorf("fetching %s: %w", url, err)`.
- Handle errors at the appropriate level — don't just propagate blindly.

### Testing

- **Test-driven**: write tests before implementation.
- **Table-driven tests** for Go — use `[]struct{ name string; ... }` pattern.
- Run `make test` after every change. No regressions allowed.
- Browser tests (go-rod) skip in `-short` mode.

### Configuration

- No hardcoded values. Configurable limits, thresholds, and sizes go in config TOML.
- Config loading: `default.toml` → `{RUN_MODE}.toml` → `APP__*` env vars (koanf).

### Concurrency

- Use `context.Context` for cancellation throughout.
- `errgroup.Group` for goroutine lifecycle management.
- `atomic` for shared counters (e.g., crawler's `max_pages`).
- Channels for pipeline stages (crawler's 3-stage pipeline).

### Zero Values

Make zero values useful where possible. A nil `*Browser` in the fetcher means "no JS rendering available" — no special initialization needed, the fetcher falls back to static HTTP.
