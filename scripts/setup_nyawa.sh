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
# Usage:
#   bash scripts/setup_nyawa.sh
#
# Requires: git (macOS build path), Go 1.23+ (macOS build path), curl.
set -euo pipefail

REPO_URL="https://github.com/rezkyauliapratama/nyawa.git"
RELEASE_TAG="v1.0.0"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nyawa"
DEST_BIN="$DEST_DIR/nyawa"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

echo "==> Detected: $OS/$ARCH"
mkdir -p "$DEST_DIR"

if [[ -x "$DEST_BIN" ]]; then
  echo "==> Nyawa already exists at $DEST_BIN — skipping. Remove it to reinstall."
  "$DEST_BIN" --version
  exit 0
fi

if [[ "$OS" == "linux" && "$ARCH" == "amd64" ]]; then
  # ---- Option A: prebuilt release binary (Linux x86_64 only) ----
  URL="https://github.com/rezkyauliapratama/nyawa/releases/download/${RELEASE_TAG}/nyawa-linux-amd64.gz"
  echo "==> Downloading release binary: $URL"
  curl -fL -o /tmp/nyawa.gz "$URL"
  gunzip -f /tmp/nyawa.gz
  mv /tmp/nyawa "$DEST_BIN"
  chmod +x "$DEST_BIN"
else
  # ---- Option B: build from source (macOS, or any other platform) ----
  echo "==> No prebuilt binary for $OS/$ARCH — building from source (tag $RELEASE_TAG)..."
  command -v git >/dev/null || { echo "ERROR: git required"; exit 1; }
  command -v go >/dev/null || { echo "ERROR: Go 1.23+ required (https://go.dev/dl/)"; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --depth 1 --branch "$RELEASE_TAG" "$REPO_URL" "$TMP/nyawa"
  (cd "$TMP/nyawa" && \
   GOOS="$OS" GOARCH="$ARCH" go build -tags "sqlite_fts5" -ldflags="-s -w" -o "$DEST_BIN" ./cmd/nyawa/)
  chmod +x "$DEST_BIN"
fi

echo "==> Verifying..."
"$DEST_BIN" --version
echo "==> Done! Nyawa installed at $DEST_BIN"
echo "    .env defaults already point here (NYAWA_BINARY=./nyawa/nyawa)."
echo "    Restart Streamlit and tick 'Use session memory (Nyawa)' in the sidebar."
