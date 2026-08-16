#!/usr/bin/env bash
# Builds Mob Cam for Linux: PyInstaller onedir, then AppImage and .deb.
# Usage: ./build_linux.sh [version]
set -e

VERSION="${1:-0.1.0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/packaging/linux/output"

echo "==> Building Mob Cam v$VERSION for Linux"

pip install --quiet -r requirements-build.txt

rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"
pyinstaller "$ROOT_DIR/mobcam.spec" --noconfirm

mkdir -p "$OUT_DIR"
bash "$ROOT_DIR/packaging/linux/build_appimage.sh" "$VERSION" "$ROOT_DIR" "$OUT_DIR"
bash "$ROOT_DIR/packaging/linux/build_deb.sh" "$VERSION" "$ROOT_DIR" "$OUT_DIR"

echo "==> Done. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
