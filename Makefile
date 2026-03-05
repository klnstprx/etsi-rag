.PHONY: build test lint run clean templ tailwind admin dev

# Binary output directory
BIN_DIR := bin

# Build the server binary
build: admin fmt
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/server ./cmd/server

# Generate templ files
templ:
	templ generate

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
	templ generate --watch &
	tailwindcss -i internal/admin/static/input.css -o internal/admin/static/styles.css --watch &
	go run ./cmd/server

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
	rm -f coverage.out coverage.html
	rm -f data/*.db
	rm -f internal/admin/static/styles.css
	rm -f internal/admin/templates/*_templ.go

# Download dependencies
deps:
	go mod download
	go mod tidy

# Format code
fmt:
	go fmt ./...
