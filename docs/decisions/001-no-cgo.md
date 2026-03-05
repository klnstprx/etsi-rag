# ADR-001: No CGO

## Status
Accepted

## Context
SQLite is the primary database. The most popular Go driver (`mattn/go-sqlite3`) requires CGO, which complicates cross-compilation and deployment.

## Decision
Use `modernc.org/sqlite` (pure Go SQLite implementation). Build with `CGO_ENABLED=0`.

## Consequences
- Portable single-binary builds on any platform
- No C compiler needed in CI/build environment
- Slight performance overhead vs. CGO driver (negligible for our workload)
- Aligns with Go proverb: "Cgo is not Go"
