# syntax=docker/dockerfile:1

# ============================================================
# Stage 1 — build the server binary + admin assets
# ============================================================
# golang:1.26 tracks the latest 1.26.x patch (>= the go.mod toolchain).
FROM golang:1.26-bookworm AS builder

# Pinned tool versions. templ matches go.mod; tailwind matches the dev machine.
ARG TEMPL_VERSION=v0.3.1118
ARG TAILWIND_VERSION=v4.3.5

WORKDIR /src

# templ CLI (.templ -> _templ.go). Installed into GOBIN (/go/bin, already on PATH).
RUN go install github.com/a-h/templ/cmd/templ@${TEMPL_VERSION}

# tailwindcss standalone CLI (no Node.js).
RUN curl -fsSL -o /usr/local/bin/tailwindcss \
        "https://github.com/tailwindlabs/tailwindcss/releases/download/${TAILWIND_VERSION}/tailwindcss-linux-x64" \
    && chmod +x /usr/local/bin/tailwindcss

# Cache module downloads before copying the rest of the source.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Mirror `make build`: generate templ, build Tailwind, compile a static binary.
# CGO_ENABLED=0 keeps the binary portable (modernc.org/sqlite is pure Go).
# -buildvcs=false because .git is excluded from the build context.
RUN templ generate \
    && tailwindcss -i internal/admin/static/input.css -o internal/admin/static/styles.css --minify \
    && CGO_ENABLED=0 go build -trimpath -buildvcs=false -o /out/server ./cmd/server

# ============================================================
# Stage 2 — runtime
# ============================================================
# Debian slim (not scratch/alpine): the crawler shells out to poppler and
# drives a system Chromium via chromedp.
FROM debian:bookworm-slim AS runtime

# poppler-utils -> pdftotext / pdfinfo / pdftoppm (PDF text, metadata, page renders)
# chromium      -> headless browser for JS-rendered pages and screenshots (chromedp)
# fonts-liberation -> baseline fonts so Chromium renders text in screenshots
# ca-certificates  -> outbound TLS to the crawl targets and the LLM backend
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        chromium \
        fonts-liberation \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. Inside a container the crawler starts Chromium with
# --no-sandbox (container runtimes commonly block the namespaces its sandbox
# needs), so the non-root user is what confines the browser.
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app

WORKDIR /app

# Binary + the tracked config templates (config.toml is generated on first run).
# Admin assets (styles.css, htmx) and prompts are //go:embed'd into the binary.
COPY --from=builder /out/server /app/server
COPY --chown=app:app config/ /app/config/
COPY --chown=app:app docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/data \
    && chown -R app:app /app

USER app

# SQLite DB + WAL/SHM live here; mount a volume to persist the corpus.
VOLUME ["/app/data"]

# Default server.port (override via APP__SERVER__PORT).
EXPOSE 3000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/server"]
