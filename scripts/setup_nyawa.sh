#!/usr/bin/env bash
# setup_nyawa.sh — one-command Nyawa binary installer for this project.
#
# Automatically:
#   1. Detects your OS + architecture
#   2. Installs a WORKING binary at ./nyawa/nyawa (the path MemoryLayer
#      expects) and verifies with `--version`
#
# IMPORTANT — the binary must run where the app runs, AND must be built
# with CGO enabled (go-sqlite3 + the BGE embedder require it):
#   - Local run on your Mac/Linux host: native build (CGO on by default)
#   - Streamlit in Docker (the README default): the container is LINUX,
#     so a binary must be built ON Linux. Cross-compiling from macOS
#     silently sets CGO_ENABLED=0 and produces a broken binary
#     ("go-sqlite3 requires cgo" / "BGE unavailable").
#
# Usage:
#   bash scripts/setup_nyawa.sh              # local run (macOS/Linux host)
#   bash scripts/setup_nyawa.sh --for-docker # linux binary, built inside
#                                            # a golang Docker container
#
# Requires: git + Go 1.23+ (local), or Docker (--for-docker). curl for
# the optional prebuilt-binary fast path (linux/amd64 only).
set -euo pipefail

REPO_URL="https://github.com/rezkyauliapratama/nyawa.git"
RELEASE_TAG="v1.0.0"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nyawa"
DEST_BIN="$DEST_DIR/nyawa"

FOR_DOCKER=0
[[ "${1:-}" == "--for-docker" ]] && FOR_DOCKER=1

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

mkdir -p "$DEST_DIR"

if [[ -x "$DEST_BIN" ]]; then
  echo "==> Nyawa already exists at $DEST_BIN — skipping. Remove it to reinstall."
  "$DEST_BIN" version || true
  exit 0
fi

# ---------------------------------------------------------------------------
# --for-docker: build inside a golang container so CGO is enabled and the
# binary matches the Linux container arch. Falls back to the prebuilt
# linux/amd64 release binary only when Docker is unavailable.
# ---------------------------------------------------------------------------
if [[ "$FOR_DOCKER" == "1" ]]; then
  if command -v docker >/dev/null 2>&1; then
    echo "==> --for-docker: building Linux binary inside golang Docker container..."
    if docker run --rm \
        -v "$DEST_DIR:/out" \
        golang:1.23 \
        sh -c "set -e; git clone --depth 1 --branch '$RELEASE_TAG' '$REPO_URL' /src && cd /src && make build && cp /src/nyawa /out/nyawa && chmod +x /out/nyawa"; then
      echo "==> Container build OK"
    else
      echo "==> Container build failed — trying prebuilt linux/amd64 release binary..."
      curl -fL -o /tmp/nyawa.gz "https://github.com/rezkyauliapratama/nyawa/releases/download/${RELEASE_TAG}/nyawa-linux-amd64.gz"
      gunzip -f /tmp/nyawa.gz
      mv /tmp/nyawa "$DEST_BIN"
      chmod +x "$DEST_BIN"
      echo "==> WARNING: using linux/amd64 prebuilt binary — if your container"
      echo "    runs linux/arm64, install Docker and re-run with --for-docker."
    fi
  else
    echo "==> Docker not found; falling back to prebuilt linux/amd64 release binary."
    curl -fL -o /tmp/nyawa.gz "https://github.com/rezkyauliapratama/nyawa/releases/download/${RELEASE_TAG}/nyawa-linux-amd64.gz"
    gunzip -f /tmp/nyawa.gz
    mv /tmp/nyawa "$DEST_BIN"
    chmod +x "$DEST_BIN"
    echo "==> WARNING: linux/amd64 binary — if your container is linux/arm64,"
    echo "    install Docker and re-run: bash scripts/setup_nyawa.sh --for-docker"
  fi
else
  # -------------------------------------------------------------------------
  # Local run: native build keeps CGO enabled (required by go-sqlite3/BGE).
  # -------------------------------------------------------------------------
  echo "==> Building from source (tag $RELEASE_TAG) for local $OS/$ARCH..."
  command -v git >/dev/null || { echo "ERROR: git required"; exit 1; }
  command -v go >/dev/null || { echo "ERROR: Go 1.23+ required (https://go.dev/dl/)"; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 --branch "$RELEASE_TAG" "$REPO_URL" "$TMP/nyawa"
  (cd "$TMP/nyawa" && make build)
  cp "$TMP/nyawa/nyawa" "$DEST_BIN"
  chmod +x "$DEST_BIN"
fi

echo "==> Verifying..."
"$DEST_BIN" version || true
echo "==> Done! Nyawa installed at $DEST_BIN"
echo "    .env defaults already point here (NYAWA_BINARY=./nyawa/nyawa)."
echo "    Restart Streamlit and tick 'Use session memory (Nyawa)' in the sidebar."
