.PHONY: build test lint run clean reset-db templ tailwind admin dev

# Binary output directory
BIN_DIR := bin

# SQLite database file. WAL/SHM siblings live next to it.
DB_FILE := data/etsi-rag.db

# Build the server binary
build: admin fmt
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/server ./cmd/server

# templ generator version, pinned to the runtime in go.mod: generated code
# from a different CLI version does not compile against it.
TEMPL_VERSION := $(shell go list -m -f '{{.Version}}' github.com/a-h/templ)
TEMPL := go run github.com/a-h/templ/cmd/templ@$(TEMPL_VERSION)

# Generate templ files
templ:
	$(TEMPL) generate

# Build Tailwind CSS
tailwind:
	tailwindcss -i internal/admin/static/input.css -o internal/admin/static/styles.css --minify

# Build admin assets (templ + tailwind)
admin: templ tailwind

# Run all tests
test:
	go test -race -v ./...

# Run tests with coverage
test-coverage:
	go test -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Run linter (requires golangci-lint)
lint:
	golangci-lint run ./...

# Run the server
run: build
	./$(BIN_DIR)/server

# Development mode with watchers
dev:
	$(TEMPL) generate --watch &
	tailwindcss -i internal/admin/static/input.css -o internal/admin/static/styles.css --watch &
	go run ./cmd/server

# Clean build artifacts. Deliberately does NOT touch the SQLite DB — rebuilding
# a corpus takes much longer than rebuilding the binary. Use `make reset-db`
# when a schema change actually requires it.
clean:
	rm -rf $(BIN_DIR)
	rm -f coverage.out coverage.html
	rm -f internal/admin/static/styles.css
	rm -f internal/admin/templates/*_templ.go

# Rename the SQLite DB (and its WAL/SHM siblings) to a timestamped .bak so a
# fresh schema migrates on the next server start. The backup is recoverable
# until you delete it manually. Run this after a schema change instead of
# `make clean`.
reset-db:
	@ts=$$(date +%s); moved=0; \
	for f in $(DB_FILE) $(DB_FILE)-shm $(DB_FILE)-wal; do \
		if [ -f "$$f" ]; then \
			mv "$$f" "$$f.bak.$$ts"; \
			moved=1; \
		fi; \
	done; \
	if [ $$moved -eq 1 ]; then \
		echo "DB renamed to $(DB_FILE).bak.$$ts* — next server start will migrate a fresh one."; \
		echo "Recover with: mv $(DB_FILE).bak.$$ts $(DB_FILE) (and the -shm/-wal siblings if present)."; \
	else \
		echo "No DB at $(DB_FILE) — nothing to reset."; \
	fi

# Download dependencies
deps:
	go mod download
	go mod tidy

# Format code
fmt:
	go fmt ./...
