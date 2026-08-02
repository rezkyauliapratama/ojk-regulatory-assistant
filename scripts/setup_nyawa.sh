#!/usr/bin/env bash
# setup_nyawa.sh — one-command Nyawa binary installer for this project.
#
# Automatically:
#   1. Detects your OS + architecture
#   2. Linux x86_64  -> downloads the official v1.0.0 release binary
#   3. macOS (arm64/amd64) -> clones the v1.0.0 tag and builds from source
#      (no prebuilt darwin release exists yet)
#   4. Places the binary at ./nyawa/nyawa (the path MemoryLayer expects)
#   5. Verifies with `--version`
#
# IMPORTANT: the binary must match where the app runs.
# - Streamlit in Docker (default per README) -> the container is LINUX,
#   so pass --for-docker to build a linux binary (macOS binary won't run).
# - Streamlit run locally on your Mac -> no flag needed (darwin binary).
#
# Usage:
#   bash scripts/setup_nyawa.sh              # for local macOS/Linux run
#   bash scripts/setup_nyawa.sh --for-docker # linux binary for the container
#
# Requires: git (build path), Go 1.23+ (build path), curl (linux release).
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

# Docker containers are linux; force target OS when requested
TARGET_OS="$OS"
if [[ "$FOR_DOCKER" == "1" ]]; then
  TARGET_OS="linux"
  echo "==> --for-docker: building/using a LINUX binary for the container"
fi

echo "==> Detected: $OS/$ARCH -> target: $TARGET_OS/$ARCH"
mkdir -p "$DEST_DIR"

if [[ -x "$DEST_BIN" ]]; then
  echo "==> Nyawa already exists at $DEST_BIN — skipping. Remove it to reinstall."
  "$DEST_BIN" --version || true
  exit 0
fi

if [[ "$TARGET_OS" == "linux" && "$ARCH" == "amd64" && "$FOR_DOCKER" != "1" ]]; then
  # ---- Option A: prebuilt release binary (host Linux x86_64 only) ----
  URL="https://github.com/rezkyauliapratama/nyawa/releases/download/${RELEASE_TAG}/nyawa-linux-amd64.gz"
  echo "==> Downloading release binary: $URL"
  curl -fL -o /tmp/nyawa.gz "$URL"
  gunzip -f /tmp/nyawa.gz
  mv /tmp/nyawa "$DEST_BIN"
  chmod +x "$DEST_BIN"
else
  # ---- Option B: build from source (macOS host, or --for-docker) ----
  echo "==> No suitable prebuilt binary — building from source (tag $RELEASE_TAG)..."
  command -v git >/dev/null || { echo "ERROR: git required"; exit 1; }
  command -v go >/dev/null || { echo "ERROR: Go 1.23+ required (https://go.dev/dl/)"; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 --branch "$RELEASE_TAG" "$REPO_URL" "$TMP/nyawa"
  (cd "$TMP/nyawa" && \
   GOOS="$TARGET_OS" GOARCH="$ARCH" go build -tags "sqlite_fts5" -ldflags="-s -w" -o "$DEST_BIN" ./cmd/nyawa/)
  chmod +x "$DEST_BIN"
fi

echo "==> Verifying..."
"$DEST_BIN" --version || true
echo "==> Done! Nyawa installed at $DEST_BIN"
echo "    .env defaults already point here (NYAWA_BINARY=./nyawa/nyawa)."
echo "    Restart Streamlit and tick 'Use session memory (Nyawa)' in the sidebar."
