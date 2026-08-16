#!/usr/bin/env bash
# Packages the PyInstaller onedir build into a Linux AppImage.
set -e

VERSION="$1"
ROOT_DIR="$2"
OUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPDIR="$SCRIPT_DIR/build/MobCam.AppDir"
APPIMAGETOOL="$SCRIPT_DIR/build/appimagetool.AppImage"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -r "$ROOT_DIR/dist/mobcam/." "$APPDIR/usr/bin/"
cp "$SCRIPT_DIR/mobcam.desktop" "$APPDIR/usr/share/applications/mobcam.desktop"
cp "$SCRIPT_DIR/mobcam.desktop" "$APPDIR/mobcam.desktop"
cp "$ROOT_DIR/logo.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/mobcam.png"
cp "$ROOT_DIR/logo.png" "$APPDIR/mobcam.png"

cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "${0}")")"
exec "${HERE}/usr/bin/mobcam" "$@"
EOF
chmod +x "$APPDIR/AppRun"

mkdir -p "$SCRIPT_DIR/build"
if [ ! -f "$APPIMAGETOOL" ]; then
  echo "==> Downloading appimagetool"
  curl -L -o "$APPIMAGETOOL" \
    https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$APPIMAGETOOL"
fi

mkdir -p "$OUT_DIR"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUT_DIR/MobCam-$VERSION-x86_64.AppImage"

echo "==> AppImage built: $OUT_DIR/MobCam-$VERSION-x86_64.AppImage"
