#!/usr/bin/env bash
# Converts logo.png into the .icns PyInstaller's BUNDLE() needs for the dock icon.
# macOS only - relies on sips and iconutil, both built into the OS.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT_DIR/logo.png"
ICONSET="$ROOT_DIR/packaging/macos/mobcam.iconset"
ICNS="$ROOT_DIR/packaging/macos/mobcam.icns"

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

for size in 16 32 64 128 256 512; do
  sips -z "$size" "$size" "$SRC" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null
done

iconutil -c icns "$ICONSET" -o "$ICNS"
echo "==> icon built: $ICNS"