#!/bin/sh
# Ensure a config.toml exists before the server starts. config.Load reads
# config/config.toml directly (it does NOT fall back to *.default.toml at
# runtime), so seed it from the template on first boot — mirroring the
# EnsureConfig copy the binary does not perform itself.
#
# A bind-mounted config.toml (e.g. config.docker.toml from docker-compose)
# already satisfies this check and is left untouched.
set -e

CONFIG_DIR="${ETSI_CONFIG_DIR:-config}"

if [ ! -f "${CONFIG_DIR}/config.toml" ]; then
    echo "config.toml not found — seeding from config.default.toml" >&2
    cp "${CONFIG_DIR}/config.default.toml" "${CONFIG_DIR}/config.toml"
fi

if [ ! -f "${CONFIG_DIR}/sources.toml" ] && [ -f "${CONFIG_DIR}/sources.default.toml" ]; then
    cp "${CONFIG_DIR}/sources.default.toml" "${CONFIG_DIR}/sources.toml"
fi

exec "$@"
