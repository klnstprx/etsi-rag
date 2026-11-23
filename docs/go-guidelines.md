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

Keep interfaces small and focused. The `llm.Embedder` interface exists for testability — it wraps only what consumers need. The `tokencount.Counter` interface has two implementations (LlamaCpp, Tiktoken) selected at runtime. The agent takes `*llm.ChatClient` concretely because it needs `ChatWithTools`.

```go
// Good: small, focused
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}
```

### Error Handling

- Return errors, don't panic.
- Wrap errors with context: `fmt.Errorf("fetching %s: %w", url, err)`.
- Handle errors at the appropriate level — don't just propagate blindly.

### Testing

*Principles from [Learn Go with Tests](https://quii.gitbook.io/learn-go-with-tests).*

#### TDD Cycle

Follow red-green-refactor strictly:

1. **Write a small test**, see it fail with a clear error message
2. **Write the minimum code** to make it pass
3. **Refactor** — do not change behavior while refactoring

Stay green. If you're red for too long, you've taken too big a step. Back up and take a smaller one.

#### Test Behavior, Not Implementation

- Tests should verify **what** code does (public API), not **how** it does it internally.
- If refactoring requires changing tests, those tests are coupled to implementation details.
- Assert only on fields relevant to the scenario — don't check entire structs when one field matters.
- One assertion per test/subtest. Use `t.Run()` to separate concerns.

#### Table-Driven Tests

Use `[]struct{ name string; ... }` for related scenarios. But don't force unrelated cases into one table — if the struct grows complex with boolean flags and conditional logic, split into separate test functions.

#### Mocking & Test Doubles

Prefer the **simplest double that works**, in this order:

1. **Real implementation** — when fast and deterministic (e.g., `*store.SQLiteStore` with `:memory:`)
2. **Fakes** — stateful in-memory implementations that mirror real behavior (preferred for complex deps)
3. **Stubs** — return canned data, stateless
4. **Spies** — record calls (couples tests to implementation — use sparingly)
5. **Mocks** — assert on expected calls (highest maintenance burden — avoid)

Rules:
- **>3 test doubles in a test = design smell.** Rethink the code, not the test.
- Don't mock the database driver and spy on SQL calls. Test against a real (in-memory) database.
- Don't make private functions public just to test them. Test through the public API.
- Difficult-to-test code is difficult-to-use code. **Listen to your tests** — they reveal design problems.

#### Dependency Injection

- Accept **interfaces**, return **structs**.
- Use stdlib interfaces where possible (`io.Writer`, `io.Reader`, `http.Handler`).
- Let callers define small interfaces for what they need — don't export large interfaces from packages.

```go
// Good: function accepts interface, testable with bytes.Buffer
func Greet(w io.Writer, name string) { fmt.Fprintf(w, "Hello, %s", name) }

// Bad: hardwired to stdout, untestable
func Greet(name string) { fmt.Printf("Hello, %s", name) }
```

#### Acceptance Tests

- Black-box tests that exercise the system through its public interface (HTTP, CLI).
- Use `testing.Short()` to skip slow acceptance tests in `-short` mode.
- Don't mock in acceptance tests — use real implementations.
- Encapsulate setup complexity in helpers; keep test logic readable.

#### Anti-Patterns to Avoid

- **Evergreen tests** — tests that can never fail are worthless. Always see the test fail first.
- **Excessive setup** — 50+ lines of setup = the code under test does too much.
- **Leaky interfaces** — interfaces with many methods force test doubles to implement unused methods. Keep interfaces to 1-2 methods.
- **Asserting irrelevant details** — testing exact error messages, log output, or internal state that isn't part of the contract.
- **Violating encapsulation** — using `package foo` instead of `package foo_test` to access unexported fields.

#### Project-Specific

- Run `make test` after every change. No regressions allowed.
- Browser tests (chromedp, need Chrome) skip in `-short` mode.
- This project uses mock structs in `_test.go` files (e.g., `mockLLM` in `pipeline_test.go`, `handler_test.go`). These are simple stubs/fakes — keep them minimal.

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
