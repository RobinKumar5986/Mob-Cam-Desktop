#!/usr/bin/env bash
# Builds Mob Cam for macOS: PyInstaller .app bundle, then wraps it into a .dmg.
# Must run on an actual Mac - PyInstaller does not cross-compile.
# Usage: ./build_macos.sh [version]
set -e

VERSION="${1:-0.1.0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/packaging/macos/output"

echo "==> Building Mob Cam v$VERSION for macOS"

python3 -m pip install --quiet -r requirements-build.txt

if [ ! -f "$ROOT_DIR/packaging/macos/mobcam.icns" ]; then
  echo "==> No icon found, generating one from logo.png"
  bash "$ROOT_DIR/packaging/macos/make_icns.sh"
fi

# Stamp the version into the spec's Info.plist before building.
sed -i '' "s/'CFBundleShortVersionString': '[^']*'/'CFBundleShortVersionString': '$VERSION'/" \
  "$ROOT_DIR/mobcam_macos.spec"

rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"
python3 -m PyInstaller "$ROOT_DIR/mobcam_macos.spec" --noconfirm

mkdir -p "$OUT_DIR"
bash "$ROOT_DIR/packaging/macos/build_dmg.sh" "$VERSION" "$ROOT_DIR" "$OUT_DIR"

echo "==> Done. Artifact in $OUT_DIR"
ls -la "$OUT_DIR"