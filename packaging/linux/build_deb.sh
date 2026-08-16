#!/usr/bin/env bash
# Packages the PyInstaller onedir build into a .deb.
set -e

VERSION="$1"
ROOT_DIR="$2"
OUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR/build/mobcam-deb"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/DEBIAN" \
         "$PKG_DIR/usr/lib/mobcam" \
         "$PKG_DIR/usr/bin" \
         "$PKG_DIR/usr/share/applications" \
         "$PKG_DIR/usr/share/icons/hicolor/256x256/apps"

cp -r "$ROOT_DIR/dist/mobcam/." "$PKG_DIR/usr/lib/mobcam/"

cat > "$PKG_DIR/usr/bin/mobcam" << 'EOF'
#!/bin/sh
exec /usr/lib/mobcam/mobcam "$@"
EOF
chmod +x "$PKG_DIR/usr/bin/mobcam"

cp "$SCRIPT_DIR/mobcam.desktop" "$PKG_DIR/usr/share/applications/mobcam.desktop"
cp "$ROOT_DIR/logo.png" "$PKG_DIR/usr/share/icons/hicolor/256x256/apps/mobcam.png"

sed "s/VERSION_PLACEHOLDER/$VERSION/" "$SCRIPT_DIR/debian/control" > "$PKG_DIR/DEBIAN/control"
cp "$SCRIPT_DIR/debian/postinst" "$PKG_DIR/DEBIAN/postinst"
chmod +x "$PKG_DIR/DEBIAN/postinst"

mkdir -p "$OUT_DIR"
DEB_FILE="$OUT_DIR/mobcam_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$PKG_DIR" "$DEB_FILE"

echo "==> .deb built: $DEB_FILE"
